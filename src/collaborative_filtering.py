"""Collaborative filtering: User-User and Item-Item CF.

Implements BOTH approaches exactly as per lecture slides:

User-User CF (personalised, normalised):
    S(u, i) = r̄_u + Σ_v∈N(u) (r_vi - r̄_v) · w_uv / Σ_v∈N(u) |w_uv|

    where w_uv = Pearson correlation:
    w_uv = Σ_i∈I_u∩I_v (r_ui - r̄_u)(r_vi - r̄_v) /
            sqrt(Σ(r_ui - r̄_u)² · Σ(r_vi - r̄_v)²)

Item-Item CF (normalised):
    S(u, i) = r̄_i + Σ_j∈N(i) (r_uj - r̄_j) · w_ij / Σ_j∈N(i) |w_ij|

    where w_ij = cosine similarity on rating vectors (adjusted cosine).
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from . import config


class ItemItemCollaborativeFiltering:
    """Item-item collaborative filtering using adjusted cosine similarity.

    From slides:
        S(u, i) = r̄_i + Σ_j∈N(i) (r_uj - r̄_j) · w_ij / Σ_j∈N(i) |w_ij|
    """

    def __init__(self, k=20, similarity="cosine"):
        self.k = k
        self.similarity = similarity
        self.user_item_matrix_ = None
        self.item_similarity_ = None
        self.user_ids_ = None
        self.item_ids_ = None
        self.user_id_to_index_ = None
        self.item_id_to_index_ = None
        self.item_means_ = None

    def fit(self, ratings):
        """Create user-item matrix and compute item-item similarities.

        Uses adjusted cosine similarity: center user ratings before computing
        cosine, so users who rate on different scales are normalised.
        """
        self.user_ids_ = np.sort(ratings[config.USER_COL].unique())
        self.item_ids_ = np.sort(ratings[config.ITEM_COL].unique())
        self.user_id_to_index_ = {
            uid: idx for idx, uid in enumerate(self.user_ids_)
        }
        self.item_id_to_index_ = {
            iid: idx for idx, iid in enumerate(self.item_ids_)
        }

        n_users = len(self.user_ids_)
        n_items = len(self.item_ids_)

        # Build dense user-item matrix (NaN for missing)
        self.user_item_matrix_ = np.full((n_users, n_items), np.nan)
        for _, row in ratings.iterrows():
            u_idx = self.user_id_to_index_[row[config.USER_COL]]
            i_idx = self.item_id_to_index_[row[config.ITEM_COL]]
            self.user_item_matrix_[u_idx, i_idx] = row[config.RATING_COL]

        # Compute per-item mean ratings (for prediction formula)
        self.item_means_ = np.nanmean(self.user_item_matrix_, axis=0)

        # Adjusted cosine: center each user's ratings by their mean
        user_means = np.nanmean(self.user_item_matrix_, axis=1, keepdims=True)
        centered = self.user_item_matrix_ - user_means
        centered = np.nan_to_num(centered, nan=0.0)

        # Item-item cosine similarity on centered matrix (columns = items)
        # centered is (n_users x n_items), we want similarity between items
        item_matrix = centered.T  # (n_items x n_users)
        norms = np.linalg.norm(item_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        item_matrix_normed = item_matrix / norms
        self.item_similarity_ = item_matrix_normed @ item_matrix_normed.T

        return self

    def predict_score(self, user_id, item_id):
        """Predict score for one user-item pair.

        From slides (Item-Item CF, normalised):
            S(u, i) = r̄_i + Σ_j∈N(i) (r_uj - r̄_j) · w_ij / Σ_j∈N(i) |w_ij|
        using top-k most similar items that the user has rated.
        """
        if user_id not in self.user_id_to_index_:
            return self.item_means_[self.item_id_to_index_.get(item_id, 0)]
        if item_id not in self.item_id_to_index_:
            return np.nanmean(self.user_item_matrix_[self.user_id_to_index_[user_id]])

        u_idx = self.user_id_to_index_[user_id]
        i_idx = self.item_id_to_index_[item_id]

        # Get items rated by this user
        user_ratings = self.user_item_matrix_[u_idx]
        rated_mask = ~np.isnan(user_ratings)
        rated_indices = np.where(rated_mask)[0]

        if len(rated_indices) == 0:
            return self.item_means_[i_idx]

        # Get similarities between target item and rated items
        sims = self.item_similarity_[i_idx, rated_indices]

        # Select top-k most similar (by absolute similarity)
        if len(sims) > self.k:
            top_k_idx = np.argsort(np.abs(sims))[-self.k:]
        else:
            top_k_idx = np.arange(len(sims))

        top_sims = sims[top_k_idx]
        top_item_indices = rated_indices[top_k_idx]
        top_ratings = user_ratings[top_item_indices]

        # Formula: S(u,i) = r̄_i + Σ_j (r_uj - r̄_j) · w_ij / Σ_j |w_ij|
        denom = np.sum(np.abs(top_sims))
        if denom == 0:
            return self.item_means_[i_idx]

        top_item_means = self.item_means_[top_item_indices]
        numerator = np.sum((top_ratings - top_item_means) * top_sims)

        return self.item_means_[i_idx] + numerator / denom

    def recommend(self, user_id, ratings_train, n=10, exclude_seen=True):
        """Generate top-n recommendations for a user."""
        if user_id not in self.user_id_to_index_:
            return []

        seen = set(
            ratings_train.loc[
                ratings_train[config.USER_COL] == user_id, config.ITEM_COL
            ]
        )

        scores = []
        candidates = self.item_ids_ if not exclude_seen else [
            i for i in self.item_ids_ if i not in seen
        ]

        for iid in candidates:
            score = self.predict_score(user_id, iid)
            scores.append((iid, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [iid for iid, _ in scores[:n]]


class UserUserCollaborativeFiltering:
    """User-user collaborative filtering using Pearson correlation.

    From slides (personalised, normalised):
        S(u, i) = r̄_u + Σ_v∈N(u) (r_vi - r̄_v) · w_uv / Σ_v∈N(u) |w_uv|

    Similarity:
        w_uv = Σ_i∈I_u∩I_v (r_ui - r̄_u)(r_vi - r̄_v) /
                sqrt(Σ(r_ui - r̄_u)² · Σ(r_vi - r̄_v)²)
    """

    def __init__(self, k=20, similarity="pearson"):
        self.k = k
        self.similarity = similarity
        self.user_item_matrix_ = None
        self.user_similarity_ = None
        self.user_ids_ = None
        self.item_ids_ = None
        self.user_id_to_index_ = None
        self.item_id_to_index_ = None
        self.user_means_ = None

    def fit(self, ratings):
        """Build user-item matrix and compute user-user Pearson correlations."""
        self.user_ids_ = np.sort(ratings[config.USER_COL].unique())
        self.item_ids_ = np.sort(ratings[config.ITEM_COL].unique())
        self.user_id_to_index_ = {
            uid: idx for idx, uid in enumerate(self.user_ids_)
        }
        self.item_id_to_index_ = {
            iid: idx for idx, iid in enumerate(self.item_ids_)
        }

        n_users = len(self.user_ids_)
        n_items = len(self.item_ids_)

        # Build dense user-item matrix
        self.user_item_matrix_ = np.full((n_users, n_items), np.nan)
        for _, row in ratings.iterrows():
            u_idx = self.user_id_to_index_[row[config.USER_COL]]
            i_idx = self.item_id_to_index_[row[config.ITEM_COL]]
            self.user_item_matrix_[u_idx, i_idx] = row[config.RATING_COL]

        # Per-user mean ratings
        self.user_means_ = np.nanmean(self.user_item_matrix_, axis=1)

        # Compute Pearson correlation between all user pairs
        # Pearson = cosine on mean-centered vectors (ignoring NaN)
        n = n_users
        self.user_similarity_ = np.zeros((n, n))
        centered = self.user_item_matrix_.copy()
        for u in range(n):
            centered[u] -= self.user_means_[u]

        for u in range(n):
            for v in range(u, n):
                # Find co-rated items
                mask = ~np.isnan(self.user_item_matrix_[u]) & ~np.isnan(
                    self.user_item_matrix_[v]
                )
                if mask.sum() < 2:
                    self.user_similarity_[u, v] = 0
                    self.user_similarity_[v, u] = 0
                    continue

                cu = centered[u, mask]
                cv = centered[v, mask]
                denom = np.sqrt(np.sum(cu ** 2) * np.sum(cv ** 2))
                if denom == 0:
                    sim = 0
                else:
                    sim = np.sum(cu * cv) / denom
                self.user_similarity_[u, v] = sim
                self.user_similarity_[v, u] = sim

        return self

    def predict_score(self, user_id, item_id):
        """Predict score from slides formula:
        S(u, i) = r̄_u + Σ_v∈N(u) (r_vi - r̄_v) · w_uv / Σ_v∈N(u) |w_uv|
        """
        if user_id not in self.user_id_to_index_:
            return 3.0
        if item_id not in self.item_id_to_index_:
            return self.user_means_[self.user_id_to_index_[user_id]]

        u_idx = self.user_id_to_index_[user_id]
        i_idx = self.item_id_to_index_[item_id]

        # Find users who rated this item
        item_ratings = self.user_item_matrix_[:, i_idx]
        rated_mask = ~np.isnan(item_ratings)
        rated_mask[u_idx] = False  # exclude the target user
        rater_indices = np.where(rated_mask)[0]

        if len(rater_indices) == 0:
            return self.user_means_[u_idx]

        # Get similarities to target user
        sims = self.user_similarity_[u_idx, rater_indices]

        # Select top-k most similar neighbours
        if len(sims) > self.k:
            top_k_idx = np.argsort(np.abs(sims))[-self.k:]
        else:
            top_k_idx = np.arange(len(sims))

        top_sims = sims[top_k_idx]
        top_user_indices = rater_indices[top_k_idx]

        denom = np.sum(np.abs(top_sims))
        if denom == 0:
            return self.user_means_[u_idx]

        # S(u,i) = r̄_u + Σ (r_vi - r̄_v) · w_uv / Σ |w_uv|
        top_ratings = item_ratings[top_user_indices]
        top_means = self.user_means_[top_user_indices]
        numerator = np.sum((top_ratings - top_means) * top_sims)

        return self.user_means_[u_idx] + numerator / denom

    def recommend(self, user_id, ratings_train, n=10, exclude_seen=True):
        """Generate top-n recommendations for a user."""
        if user_id not in self.user_id_to_index_:
            return []

        seen = set(
            ratings_train.loc[
                ratings_train[config.USER_COL] == user_id, config.ITEM_COL
            ]
        )

        candidates = [i for i in self.item_ids_ if i not in seen] if exclude_seen else list(self.item_ids_)

        scores = []
        for iid in candidates:
            score = self.predict_score(user_id, iid)
            scores.append((iid, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [iid for iid, _ in scores[:n]]
