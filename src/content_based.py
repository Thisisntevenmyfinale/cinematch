"""Content-based recommender using TF-IDF item features and cosine similarity.

Follows ContentBasedFiltering.pdf exactly:

Item representation -- TF-IDF (Folie 37):
    tfidf(term, doc) = tf(term, doc) * log(N / df(term))

User profile -- implicit from centred ratings (Folien 27-28):
    profile(u) = Sigma_i (r_ui - r_u) * vector(i)

Prediction -- cosine similarity (Folie 33/43):
    score(u, i) = cos(profile_u, vector_i)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import issparse, spmatrix
from . import config


class ContentBasedRecommender:
    """Content-based recommender using item genres (TF-IDF) and user profiles.

    ContentBasedFiltering.pdf, Folien 27-43.
    """

    def __init__(self, feature_col: str = config.GENRES_COL, use_tags: bool = False) -> None:
        self.feature_col = feature_col
        self.use_tags = use_tags
        self.vectorizer: TfidfVectorizer | None = None
        self.item_features_: spmatrix | np.ndarray | None = None
        self.item_ids_: np.ndarray | None = None
        self.item_id_to_index_: dict[int, int] | None = None

    def fit(self, ratings: pd.DataFrame, items: pd.DataFrame, tags: pd.DataFrame | None = None) -> "ContentBasedRecommender":
        """Build item feature matrix using TF-IDF on genres (Folie 37).

        Genres in MovieLens are pipe-separated (e.g. "Action|Comedy|Drama").
        We replace '|' with spaces so TfidfVectorizer treats each genre as a token.

        If use_tags=True and tags DataFrame provided, we append user-generated
        tags to enrich the feature representation (Folie 17: collective tagging).
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

    def build_user_profile(self, user_id: int, ratings_train: pd.DataFrame) -> np.ndarray:
        """Build a user profile from centered ratings and TF-IDF item vectors.

        ContentBasedFiltering.pdf, Folien 27-28:
            profile(u) = Sigma_i (r_ui - r_u) * vector(i)
        Profile is L2-normalised after construction.
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

    def recommend(self, user_id: int, ratings_train: pd.DataFrame, n: int = 10, exclude_seen: bool = True) -> list[int]:
        """Recommend items whose TF-IDF vectors are most similar to user profile.

        ContentBasedFiltering.pdf, Folie 33/43:
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

    def recommend_with_scores(self, user_id: int, ratings_train: pd.DataFrame, n: int = 10, exclude_seen: bool = True) -> list[tuple[int, float]]:
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

    def similar_items(self, item_id: int, n: int = 10) -> list[tuple[int, float]]:
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
