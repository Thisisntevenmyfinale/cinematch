"""Content-based recommender using TF-IDF item features and cosine similarity.

Follows the lecture slides exactly:
- Items modeled as TF-IDF vectors in keyword/genre space
- User profile built from centered ratings:
    profile(u) = Σ_i (r_ui - r̄_u) * vector(i)
- Prediction via cosine similarity:
    score(u, i) = cos(profile_u, vector_i)

TF-IDF from slides:
    tfidf(term, doc) = tf(term, doc) * log(N / df(term))
    where N = number of documents, df = document frequency
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import issparse
from . import config


class ContentBasedRecommender:
    """Content-based recommender using item genres (TF-IDF) and user profiles."""

    def __init__(self, feature_col=config.GENRES_COL, use_tags=False):
        self.feature_col = feature_col
        self.use_tags = use_tags
        self.vectorizer = None
        self.item_features_ = None  # TF-IDF matrix (n_items x n_features)
        self.item_ids_ = None  # array of item IDs in matrix order
        self.item_id_to_index_ = None  # dict: item_id -> row index

    def fit(self, ratings, items, tags=None):
        """Build item feature matrix using TF-IDF on genres.

        Genres in MovieLens are pipe-separated (e.g. "Action|Comedy|Drama").
        We replace '|' with spaces so TfidfVectorizer treats each genre as a token.

        If use_tags=True and tags DataFrame provided, we append user-generated
        tags to enrich the feature representation.
        """
        items_clean = items.dropna(subset=[self.feature_col]).copy()

        # Convert genres: "Action|Comedy" -> "Action Comedy"
        text_features = items_clean[self.feature_col].str.replace("|", " ", regex=False)

        # Optionally enrich with user-generated tags
        if self.use_tags and tags is not None:
            tag_text = (
                tags.groupby(config.ITEM_COL)["tag"]
                .apply(lambda x: " ".join(x.astype(str).str.lower()))
                .reindex(items_clean[config.ITEM_COL])
                .fillna("")
            )
            text_features = text_features.values + " " + tag_text.values

        # Fit TF-IDF vectorizer (as per slides: tf * log(N/df))
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            token_pattern=r"(?u)\b\w[\w\-]+\b",
        )
        self.item_features_ = self.vectorizer.fit_transform(text_features)
        self.item_ids_ = items_clean[config.ITEM_COL].values
        self.item_id_to_index_ = {
            iid: idx for idx, iid in enumerate(self.item_ids_)
        }
        return self

    def build_user_profile(self, user_id, ratings_train):
        """Build a user profile from centered ratings and TF-IDF item vectors.

        From slides:
            profile(u) = Σ_i (rating(u,i) - mean_rating(u)) * vector(i)
        """
        user_ratings = ratings_train[
            ratings_train[config.USER_COL] == user_id
        ]
        if len(user_ratings) == 0:
            n_features = self.item_features_.shape[1]
            return np.zeros(n_features)

        mean_rating = user_ratings[config.RATING_COL].mean()

        profile = np.zeros(self.item_features_.shape[1])
        for _, row in user_ratings.iterrows():
            iid = row[config.ITEM_COL]
            if iid in self.item_id_to_index_:
                idx = self.item_id_to_index_[iid]
                centered_rating = row[config.RATING_COL] - mean_rating
                vec = self.item_features_[idx]
                if issparse(vec):
                    vec = vec.toarray().flatten()
                profile += centered_rating * vec

        # Normalize profile
        norm = np.linalg.norm(profile)
        if norm > 0:
            profile = profile / norm
        return profile

    def recommend(self, user_id, ratings_train, n=10, exclude_seen=True):
        """Recommend items whose TF-IDF vectors are most similar to user profile.

        Prediction from slides:
            score(u, i) = cos(profile_u, vector_i)
        """
        profile = self.build_user_profile(user_id, ratings_train)

        # Cosine similarity between profile and all items
        profile_2d = profile.reshape(1, -1)
        scores = cosine_similarity(profile_2d, self.item_features_).flatten()

        # Build scored item list
        scored_items = list(zip(self.item_ids_, scores))

        if exclude_seen:
            seen = set(
                ratings_train.loc[
                    ratings_train[config.USER_COL] == user_id, config.ITEM_COL
                ]
            )
            scored_items = [(iid, s) for iid, s in scored_items if iid not in seen]

        # Sort by score descending
        scored_items.sort(key=lambda x: x[1], reverse=True)
        top_items = scored_items[:n]
        return [iid for iid, _ in top_items]

    def recommend_with_scores(self, user_id, ratings_train, n=10, exclude_seen=True):
        """Same as recommend but also returns scores."""
        profile = self.build_user_profile(user_id, ratings_train)
        profile_2d = profile.reshape(1, -1)
        scores = cosine_similarity(profile_2d, self.item_features_).flatten()

        scored_items = list(zip(self.item_ids_, scores))
        if exclude_seen:
            seen = set(
                ratings_train.loc[
                    ratings_train[config.USER_COL] == user_id, config.ITEM_COL
                ]
            )
            scored_items = [(iid, s) for iid, s in scored_items if iid not in seen]

        scored_items.sort(key=lambda x: x[1], reverse=True)
        return scored_items[:n]

    def similar_items(self, item_id, n=10):
        """Find items most similar to a given item (item-to-item content similarity)."""
        if item_id not in self.item_id_to_index_:
            return []
        idx = self.item_id_to_index_[item_id]
        item_vec = self.item_features_[idx]
        sims = cosine_similarity(item_vec, self.item_features_).flatten()
        # Exclude the item itself
        top_indices = np.argsort(sims)[::-1]
        result = []
        for i in top_indices:
            if self.item_ids_[i] != item_id:
                result.append((self.item_ids_[i], sims[i]))
            if len(result) >= n:
                break
        return result
