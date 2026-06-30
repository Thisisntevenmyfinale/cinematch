"""Evaluation metrics for recommender systems.

All metrics from EvaluationRecommenderSystems.pdf:

RATING PREDICTION (Folien 11-12):
    - MAE (Mean Absolute Error)
    - RMSE (Root Mean Squared Error)

TOP-N RECOMMENDATION (Folien 15-19, per-user, then averaged):
    - Precision@K  (Folie 15)
    - Recall@K     (Folie 16)
    - Hit Rate@K
    - DCG@K        (Folie 17): Sigma_i rel_i / log2(i + 1)
    - NDCG@K       (Folie 18): DCG@K / IDCG@K
    - MRR          (Folie 19): 1/rank of first relevant item

BEYOND ACCURACY (Folien 22-26):
    - Catalog Coverage (Folie 25): |recommended items| / |all items|
    - Diversity        (Folie 23): average pairwise distance
    - Novelty          (Folie 24): average self-information
    - Popularity Bias  (Folie 26): % from top-X% popular items
    - Serendipity      (Folie 26): unexpected AND relevant items
    - Fairness: performance gap between heavy and light raters
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from . import config


# ── Rating Prediction Metrics ─────────────────────────────────────

def mae(predictions: list | np.ndarray, actuals: list | np.ndarray) -> float:
    """Mean Absolute Error (Folie 11)."""
    return float(np.mean(np.abs(np.array(predictions) - np.array(actuals))))


def rmse(predictions: list | np.ndarray, actuals: list | np.ndarray) -> float:
    """Root Mean Squared Error (Folie 12)."""
    return float(np.sqrt(np.mean((np.array(predictions) - np.array(actuals)) ** 2)))


# ── Top-N Recommendation Metrics (per user) ───────────────────────

def precision_at_k(recommended_items: list, relevant_items: set, k: int = 10) -> float:
    """Precision@K (Folie 15): fraction of recommended items that are relevant."""
    rec = list(recommended_items)[:k]
    rel = set(relevant_items)
    if len(rec) == 0:
        return 0.0
    return len(set(rec) & rel) / len(rec)


def recall_at_k(recommended_items: list, relevant_items: set, k: int = 10) -> float:
    """Recall@K (Folie 16): fraction of relevant items that are recommended."""
    rec = set(list(recommended_items)[:k])
    rel = set(relevant_items)
    if len(rel) == 0:
        return 0.0
    return len(rec & rel) / len(rel)


def hit_rate_at_k(recommended_items: list, relevant_items: set, k: int = 10) -> float:
    """Hit Rate@K: 1 if at least one relevant item in top-k, else 0."""
    rec = set(list(recommended_items)[:k])
    rel = set(relevant_items)
    return 1.0 if len(rec & rel) > 0 else 0.0


def dcg_at_k(relevance_scores: list | np.ndarray, k: int = 10) -> float:
    """DCG@K (Folie 17): Sigma_i rel_i / log2(i + 1), rank i from 1."""
    rel = np.array(relevance_scores)[:k]
    if len(rel) == 0:
        return 0.0
    ranks = np.arange(1, len(rel) + 1)
    return np.sum(rel / np.log2(ranks + 1))


def ndcg_at_k(recommended_items: list, relevant_items: set, k: int = 10) -> float:
    """NDCG@K with binary relevance (Folie 18).

    1. Build binary relevance vector for recommended items
    2. Compute DCG
    3. Compute ideal DCG (all relevant items at top)
    4. Return DCG / IDCG
    """
    rec = list(recommended_items)[:k]
    rel = set(relevant_items)

    relevance = [1.0 if item in rel else 0.0 for item in rec]
    dcg = dcg_at_k(relevance, k)

    # Ideal: all relevant items come first
    n_rel = min(len(rel), k)
    ideal_relevance = [1.0] * n_rel + [0.0] * (k - n_rel)
    idcg = dcg_at_k(ideal_relevance, k)

    if idcg == 0:
        return 0.0
    return dcg / idcg


def mean_reciprocal_rank(recommended_items: list, relevant_items: set, k: int = 10) -> float:
    """MRR (Folie 19): 1/rank of first relevant item."""
    rec = list(recommended_items)[:k]
    rel = set(relevant_items)
    for rank, item in enumerate(rec, start=1):
        if item in rel:
            return 1.0 / rank
    return 0.0


# ── Beyond-Accuracy Metrics ──────────────────────────────────────

def catalog_coverage(all_recommendations: list[list], all_items: set | list) -> float:
    """Catalog coverage (Folie 25):
    |unique recommended items| / |all items in catalog|
    """
    recommended_set = set()
    for rec_list in all_recommendations:
        recommended_set.update(rec_list)
    return len(recommended_set) / len(set(all_items)) if len(all_items) > 0 else 0.0


def intra_list_diversity(recommended_items: list, item_features: np.ndarray, item_id_to_index: dict) -> float:
    """Diversity (Folie 23): average pairwise distance in recommendation list.

    diversity(L) = 1 - mean(cos_sim(i, j)) for all pairs i,j in L
    Uses item feature vectors (e.g. TF-IDF genre vectors).
    """
    indices = [item_id_to_index[i] for i in recommended_items if i in item_id_to_index]
    if len(indices) < 2:
        return 0.0
    vectors = item_features[indices]
    sim_matrix = cosine_similarity(vectors)
    n = len(indices)
    # Average pairwise similarity (upper triangle, excluding diagonal)
    total_sim = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total_sim += sim_matrix[i, j]
            count += 1
    avg_sim = total_sim / count if count > 0 else 0.0
    return 1.0 - avg_sim


def novelty_score(recommended_items: list, item_popularity: dict, n_users: int) -> float:
    """Novelty (Folie 24): average self-information.

    novelty(i) = -log2(popularity(i) / n_users)
    Higher novelty = less popular items recommended.
    """
    scores = []
    for item in recommended_items:
        pop = item_popularity.get(item, 1)
        self_info = -np.log2(pop / n_users) if pop > 0 else 0
        scores.append(self_info)
    return np.mean(scores) if scores else 0.0


def popularity_bias(recommended_items: list, item_popularity: dict, top_percent: float = 0.1) -> float:
    """Popularity bias (Folie 26):
    What percentage of recommendations come from the top X% most popular items?
    """
    if not item_popularity:
        return 0.0
    sorted_items = sorted(item_popularity.keys(), key=lambda x: item_popularity[x], reverse=True)
    n_top = max(1, int(len(sorted_items) * top_percent))
    top_items = set(sorted_items[:n_top])
    if len(recommended_items) == 0:
        return 0.0
    return sum(1 for i in recommended_items if i in top_items) / len(recommended_items)


def serendipity_score(recommended_items: list, relevant_items: set, popular_items: set) -> float:
    """Serendipity (Folie 26): unexpected AND relevant items.

    An item is serendipitous if it is relevant but NOT in the popular baseline.
    serendipity = |relevant & recommended & ~popular| / |recommended|
    """
    rec = set(recommended_items)
    rel = set(relevant_items)
    pop = set(popular_items)
    serendipitous = (rec & rel) - pop
    return len(serendipitous) / len(rec) if len(rec) > 0 else 0.0


# ── Fairness Metric ─────────────────────────────────────────────

def user_fairness(
    model,
    ratings_train: pd.DataFrame,
    ratings_test: pd.DataFrame,
    users: list | np.ndarray,
    k: int = 10,
    percentile: int = 50,
) -> dict[str, float]:
    """Fairness: performance gap between heavy and light raters.

    Splits users at the given percentile of training interaction count.
    Returns NDCG@K for each group and the absolute gap.
    A smaller gap indicates fairer treatment across user activity levels.
    """
    user_counts = ratings_train.groupby(config.USER_COL).size()
    threshold = np.percentile(user_counts.values, percentile)

    heavy_users = set(user_counts[user_counts >= threshold].index)
    light_users = set(user_counts[user_counts < threshold].index)

    def _avg_ndcg(user_subset: set) -> float:
        scores = []
        for uid in users:
            if uid not in user_subset:
                continue
            user_test = ratings_test[ratings_test[config.USER_COL] == uid]
            relevant = set(
                user_test.loc[user_test[config.RATING_COL] >= 4.0, config.ITEM_COL]
            )
            if len(relevant) == 0:
                continue
            recs = model.recommend(uid, ratings_train, n=k, exclude_seen=True)
            scores.append(ndcg_at_k(recs, relevant, k))
        return float(np.mean(scores)) if scores else 0.0

    ndcg_heavy = _avg_ndcg(heavy_users)
    ndcg_light = _avg_ndcg(light_users)

    return {
        "NDCG@K_heavy": ndcg_heavy,
        "NDCG@K_light": ndcg_light,
        "Fairness_gap": abs(ndcg_heavy - ndcg_light),
    }


# ── Full Evaluation Loop ─────────────────────────────────────────

def evaluate_model(model, ratings_train: pd.DataFrame, ratings_test: pd.DataFrame,
                   users: list | np.ndarray, items_df: pd.DataFrame | None = None,
                   k: int = 10, item_features: np.ndarray | None = None,
                   item_id_to_index: dict | None = None,
                   item_popularity: dict | None = None, n_users: int | None = None,
                   popular_items: set | None = None) -> dict[str, float]:
    """Evaluate a recommender model over a set of users.

    Returns a dict of averaged metrics.
    """
    precisions = []
    recalls = []
    ndcgs = []
    mrrs = []
    hit_rates = []
    all_recs = []
    diversities = []
    novelties = []
    pop_biases = []
    serendipities = []

    all_items = set(ratings_train[config.ITEM_COL].unique()) | set(
        ratings_test[config.ITEM_COL].unique()
    )

    for user_id in users:
        # Get relevant items: test items with rating >= 4.0
        user_test = ratings_test[ratings_test[config.USER_COL] == user_id]
        relevant = set(
            user_test.loc[
                user_test[config.RATING_COL] >= 4.0, config.ITEM_COL
            ]
        )

        if len(relevant) == 0:
            continue

        # Generate recommendations
        recs = model.recommend(user_id, ratings_train, n=k, exclude_seen=True)
        all_recs.append(recs)

        # Core metrics
        precisions.append(precision_at_k(recs, relevant, k))
        recalls.append(recall_at_k(recs, relevant, k))
        ndcgs.append(ndcg_at_k(recs, relevant, k))
        mrrs.append(mean_reciprocal_rank(recs, relevant, k))
        hit_rates.append(hit_rate_at_k(recs, relevant, k))

        # Beyond-accuracy metrics
        if item_features is not None and item_id_to_index is not None:
            diversities.append(
                intra_list_diversity(recs, item_features, item_id_to_index)
            )

        if item_popularity is not None and n_users is not None:
            novelties.append(novelty_score(recs, item_popularity, n_users))
            pop_biases.append(popularity_bias(recs, item_popularity))

        if popular_items is not None:
            serendipities.append(
                serendipity_score(recs, relevant, popular_items)
            )

    results = {
        "Precision@K": np.mean(precisions) if precisions else 0.0,
        "Recall@K": np.mean(recalls) if recalls else 0.0,
        "NDCG@K": np.mean(ndcgs) if ndcgs else 0.0,
        "MRR": np.mean(mrrs) if mrrs else 0.0,
        "Hit Rate@K": np.mean(hit_rates) if hit_rates else 0.0,
        "Coverage": catalog_coverage(all_recs, all_items) if all_recs else 0.0,
    }

    if diversities:
        results["Diversity"] = np.mean(diversities)
    if novelties:
        results["Novelty"] = np.mean(novelties)
    if pop_biases:
        results["Popularity Bias"] = np.mean(pop_biases)
    if serendipities:
        results["Serendipity"] = np.mean(serendipities)

    return results
