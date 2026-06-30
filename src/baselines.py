"""Non-personalized baseline recommenders.

Implements three baselines as per lecture slides:
- Most Popular (by interaction count)
- Highest Average Rating (with minimum rating threshold)
- Random (control baseline)

Non-personalised formula from slides:
    S(u, i) = Σ_v r_vi / |U|   (simple average of all ratings for item i)
"""

import numpy as np
import pandas as pd
from . import config


class MostPopularRecommender:
    """Recommend the most frequently rated items (popularity = interaction count)."""

    def __init__(self):
        self.ranking_ = None

    def fit(self, ratings, items=None):
        """Count interactions per item and rank by popularity."""
        popularity = (
            ratings.groupby(config.ITEM_COL)
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )
        self.ranking_ = popularity[config.ITEM_COL].values
        return self

    def recommend(self, user_id, ratings_train, n=10, exclude_seen=True):
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

    From slides (non-personalised):
        S(u, i) = Σ_v r_vi / |U|
    We filter items with fewer than min_ratings to avoid noise.
    """

    def __init__(self, min_ratings=20):
        self.min_ratings = min_ratings
        self.ranking_ = None

    def fit(self, ratings, items=None):
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

    def recommend(self, user_id, ratings_train, n=10, exclude_seen=True):
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
    """Recommend random unseen items (control baseline)."""

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.items_ = None

    def fit(self, ratings, items=None):
        """Store the list of all item IDs."""
        self.items_ = ratings[config.ITEM_COL].unique()
        return self

    def recommend(self, user_id, ratings_train, n=10, exclude_seen=True):
        """Sample n random unseen items."""
        rng = np.random.RandomState(self.random_state + hash(user_id) % 10000)
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
