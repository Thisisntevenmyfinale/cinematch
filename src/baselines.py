"""Non-personalized baseline recommenders.

Implements three baselines as per lecture slides
(CollaborativeFiltering.pdf, Folien 12-13; EvaluationRecommenderSystems.pdf, Folie 28):

- Most Popular (by interaction count) -- standard baseline for comparison
- Highest Average Rating: S(u, i) = Sigma_v r_vi / |U|  (Folie 12)
- Random (control baseline, recommended in Folie 28)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from . import config


class MostPopularRecommender:
    """Recommend the most frequently rated items (popularity = interaction count).

    Standard baseline referenced in EvaluationRecommenderSystems.pdf, Folie 28:
    'Always compare against simple baselines: Most popular'.
    """

    def __init__(self) -> None:
        self.ranking_: np.ndarray | None = None

    def fit(self, ratings: pd.DataFrame, items: pd.DataFrame | None = None) -> "MostPopularRecommender":
        """Count interactions per item and rank by popularity."""
        popularity = (
            ratings.groupby(config.ITEM_COL)
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )
        self.ranking_ = popularity[config.ITEM_COL].values
        return self

    def recommend(self, user_id: int, ratings_train: pd.DataFrame, n: int = 10, exclude_seen: bool = True) -> list[int]:
        """Return top-n most popular items not already seen by user."""
        if exclude_seen:
            seen = set(
                ratings_train.loc[
                    ratings_train[config.USER_COL] == user_id, config.ITEM_COL
                ]
            )
            candidates = [i for i in self.ranking_ if i not in seen]
        else:
            candidates = list(self.ranking_)
        return candidates[:n]


class HighestAverageRatingRecommender:
    """Recommend items with the highest average rating (min ratings filter).

    From CollaborativeFiltering.pdf, Folie 12 (non-personalised):
        S(u, i) = Sigma_v r_vi / |U|
    Items with fewer than min_ratings are filtered to avoid noise from
    low-sample items.
    """

    def __init__(self, min_ratings: int = 20) -> None:
        self.min_ratings = min_ratings
        self.ranking_: np.ndarray | None = None

    def fit(self, ratings: pd.DataFrame, items: pd.DataFrame | None = None) -> "HighestAverageRatingRecommender":
        """Compute average rating per item, filter by min_ratings, rank."""
        item_stats = ratings.groupby(config.ITEM_COL)[config.RATING_COL].agg(
            ["mean", "count"]
        )
        # Filter items with enough ratings
        qualified = item_stats[item_stats["count"] >= self.min_ratings]
        # Sort by mean rating descending, break ties by count
        qualified = qualified.sort_values(
            ["mean", "count"], ascending=[False, False]
        )
        self.ranking_ = qualified.index.values
        return self

    def recommend(self, user_id: int, ratings_train: pd.DataFrame, n: int = 10, exclude_seen: bool = True) -> list[int]:
        """Return top-n highest-rated items not already seen by user."""
        if exclude_seen:
            seen = set(
                ratings_train.loc[
                    ratings_train[config.USER_COL] == user_id, config.ITEM_COL
                ]
            )
            candidates = [i for i in self.ranking_ if i not in seen]
        else:
            candidates = list(self.ranking_)
        return candidates[:n]


class RandomRecommender:
    """Recommend random unseen items (control baseline).

    Referenced in EvaluationRecommenderSystems.pdf, Folie 28:
    'Always compare against simple baselines: Random'.
    """

    def __init__(self, random_state: int = config.RANDOM_STATE) -> None:
        self.random_state = random_state
        self.items_: np.ndarray | None = None

    def fit(self, ratings: pd.DataFrame, items: pd.DataFrame | None = None) -> "RandomRecommender":
        """Store the list of all item IDs."""
        self.items_ = ratings[config.ITEM_COL].unique()
        return self

    def recommend(self, user_id: int, ratings_train: pd.DataFrame, n: int = 10, exclude_seen: bool = True) -> list[int]:
        """Sample n random unseen items."""
        rng = np.random.RandomState(self.random_state + hash(user_id) % 10_000)
        if exclude_seen:
            seen = set(
                ratings_train.loc[
                    ratings_train[config.USER_COL] == user_id, config.ITEM_COL
                ]
            )
            candidates = np.array([i for i in self.items_ if i not in seen])
        else:
            candidates = self.items_.copy()
        if len(candidates) <= n:
            return list(candidates)
        chosen = rng.choice(candidates, size=n, replace=False)
        return list(chosen)
