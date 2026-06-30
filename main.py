"""Main pipeline for the Movie Recommender System.

Orchestrates: Data loading → EDA → Train/Test Split → Model Training →
Evaluation → Comparison → Recommendation Examples.

Dataset: MovieLens Latest Small (100,836 ratings, 9,742 movies, 610 users)
Source: https://grouplens.org/datasets/movielens/latest/
License: See data/raw/README.txt for full GroupLens license terms.
Citation: F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens
          Datasets: History and Context. ACM TIKDDExplor. Newsl. 19, 4.
"""

import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import config
from src.data_loading import (
    load_ratings, load_items, load_tags,
    train_test_split_ratings, describe_dataset, get_seen_items,
)
from src.baselines import (
    MostPopularRecommender, HighestAverageRatingRecommender, RandomRecommender,
)
from src.content_based import ContentBasedRecommender
from src.collaborative_filtering import (
    ItemItemCollaborativeFiltering, UserUserCollaborativeFiltering,
)
from src.matrix_factorization import MatrixFactorizationRecommender
from src.evaluation import evaluate_model, mae, rmse

warnings.filterwarnings("ignore", category=FutureWarning)


def main():
    print("=" * 60)
    print("  MOVIE RECOMMENDER SYSTEM — MovieLens Latest Small")
    print("=" * 60)

    # ── 1. Load Data ──────────────────────────────────────────
    print("\n[1/7] Loading dataset...")
    ratings = load_ratings()
    items = load_items()
    tags = load_tags()
    print(f"  Loaded {len(ratings):,} ratings, {len(items):,} items, {len(tags):,} tags")

    # ── 2. EDA ────────────────────────────────────────────────
    print("\n[2/7] Exploratory Data Analysis")
    stats = describe_dataset(ratings, items)

    # Save EDA figures
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Rating distribution
    ratings[config.RATING_COL].hist(bins=10, ax=axes[0], edgecolor="black")
    axes[0].set_title("Rating Distribution")
    axes[0].set_xlabel("Rating")
    axes[0].set_ylabel("Count")

    # Ratings per user distribution
    ratings.groupby(config.USER_COL).size().hist(bins=50, ax=axes[1], edgecolor="black")
    axes[1].set_title("Ratings per User")
    axes[1].set_xlabel("Number of Ratings")
    axes[1].set_ylabel("Number of Users")

    # Ratings per item distribution
    ratings.groupby(config.ITEM_COL).size().hist(bins=50, ax=axes[2], edgecolor="black")
    axes[2].set_title("Ratings per Item")
    axes[2].set_xlabel("Number of Ratings")
    axes[2].set_ylabel("Number of Items")

    plt.tight_layout()
    plt.savefig(config.RESULTS_DIR / "figures" / "eda_distributions.png", dpi=150)
    plt.close()
    print("  Saved EDA figures to results/figures/eda_distributions.png")

    # ── 3. Train/Test Split ───────────────────────────────────
    print("\n[3/7] Creating train/test split (80/20, stratified by user)...")
    train, test = train_test_split_ratings(ratings, test_size=0.2)
    print(f"  Train: {len(train):,} ratings | Test: {len(test):,} ratings")

    # Precompute helpers for evaluation
    item_popularity = train.groupby(config.ITEM_COL).size().to_dict()
    n_users = train[config.USER_COL].nunique()
    # Top 10% most popular items (for popularity bias & serendipity)
    sorted_by_pop = sorted(item_popularity.keys(), key=lambda x: item_popularity[x], reverse=True)
    n_top = max(1, int(len(sorted_by_pop) * 0.1))
    popular_items = set(sorted_by_pop[:n_top])

    # Users to evaluate (all users that have at least one test rating >= 4.0)
    test_relevant = test[test[config.RATING_COL] >= 4.0]
    eval_users = test_relevant[config.USER_COL].unique()
    print(f"  Users with relevant test items: {len(eval_users)}")

    # ── 4. Train Models ──────────────────────────────────────
    print("\n[4/7] Training recommender models...")
    models = {}
    timings = {}

    # 4a. Random Baseline
    t0 = time.time()
    random_rec = RandomRecommender(random_state=42)
    random_rec.fit(train)
    timings["Random"] = time.time() - t0
    models["Random"] = random_rec
    print(f"  Random Baseline: trained ({timings['Random']:.2f}s)")

    # 4b. Most Popular
    t0 = time.time()
    pop_rec = MostPopularRecommender()
    pop_rec.fit(train)
    timings["Most Popular"] = time.time() - t0
    models["Most Popular"] = pop_rec
    print(f"  Most Popular: trained ({timings['Most Popular']:.2f}s)")

    # 4c. Highest Average
    t0 = time.time()
    avg_rec = HighestAverageRatingRecommender(min_ratings=20)
    avg_rec.fit(train)
    timings["Highest Avg"] = time.time() - t0
    models["Highest Avg"] = avg_rec
    print(f"  Highest Average: trained ({timings['Highest Avg']:.2f}s)")

    # 4d. Content-Based (TF-IDF on genres)
    t0 = time.time()
    cb_rec = ContentBasedRecommender(use_tags=False)
    cb_rec.fit(train, items)
    timings["Content-Based"] = time.time() - t0
    models["Content-Based"] = cb_rec
    print(f"  Content-Based (TF-IDF genres): trained ({timings['Content-Based']:.2f}s)")

    # 4e. Content-Based with Tags
    t0 = time.time()
    cb_tags_rec = ContentBasedRecommender(use_tags=True)
    cb_tags_rec.fit(train, items, tags=tags)
    timings["CB + Tags"] = time.time() - t0
    models["CB + Tags"] = cb_tags_rec
    print(f"  Content-Based (TF-IDF + Tags): trained ({timings['CB + Tags']:.2f}s)")

    # 4f. Item-Item CF
    t0 = time.time()
    ii_cf = ItemItemCollaborativeFiltering(k=30, similarity="cosine")
    ii_cf.fit(train)
    timings["Item-Item CF"] = time.time() - t0
    models["Item-Item CF"] = ii_cf
    print(f"  Item-Item CF: trained ({timings['Item-Item CF']:.2f}s)")

    # 4g. User-User CF (note: O(n^2) — may take longer)
    print("  User-User CF: training (this may take a few minutes)...")
    t0 = time.time()
    uu_cf = UserUserCollaborativeFiltering(k=30, similarity="pearson")
    uu_cf.fit(train)
    timings["User-User CF"] = time.time() - t0
    models["User-User CF"] = uu_cf
    print(f"  User-User CF: trained ({timings['User-User CF']:.2f}s)")

    # 4h. Matrix Factorization (Biased SGD)
    t0 = time.time()
    mf_rec = MatrixFactorizationRecommender(
        n_factors=50, n_epochs=20, lr=0.005, reg=0.02, verbose=True
    )
    mf_rec.fit(train)
    timings["Matrix Fact."] = time.time() - t0
    models["Matrix Fact."] = mf_rec
    print(f"  Matrix Factorization: trained ({timings['Matrix Fact.']:.2f}s)")

    # ── 5. Evaluate All Models ────────────────────────────────
    print("\n[5/7] Evaluating all models...")
    K = config.TOP_K
    results = {}

    # Get content-based item features for diversity calculation
    cb_features = cb_rec.item_features_
    cb_id_to_idx = cb_rec.item_id_to_index_

    for name, model in models.items():
        print(f"  Evaluating {name}...")
        res = evaluate_model(
            model, train, test, eval_users, items_df=items, k=K,
            item_features=cb_features, item_id_to_index=cb_id_to_idx,
            item_popularity=item_popularity, n_users=n_users,
            popular_items=popular_items,
        )
        res["Training Time (s)"] = timings.get(name, 0)
        results[name] = res

    # ── 6. Results Table ──────────────────────────────────────
    print("\n[6/7] Results Comparison")
    results_df = pd.DataFrame(results).T
    results_df.index.name = "Model"

    # Format nicely
    pd.set_option("display.max_columns", 15)
    pd.set_option("display.width", 120)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print("\n" + "=" * 100)
    print(f"EVALUATION RESULTS (K={K})")
    print("=" * 100)
    print(results_df.to_string())
    print("=" * 100)

    # Save results
    results_df.to_csv(config.RESULTS_DIR / "metrics.csv")
    print(f"\n  Saved metrics to results/metrics.csv")

    # Visualisation: metrics comparison bar chart
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    metrics_to_plot = ["Precision@K", "Recall@K", "NDCG@K", "Diversity", "Novelty", "Coverage"]
    for ax, metric in zip(axes.flatten(), metrics_to_plot):
        if metric in results_df.columns:
            results_df[metric].plot(kind="barh", ax=ax)
            ax.set_title(metric)
            ax.set_xlabel("Score")
    plt.tight_layout()
    plt.savefig(config.RESULTS_DIR / "figures" / "model_comparison.png", dpi=150)
    plt.close()
    print("  Saved comparison chart to results/figures/model_comparison.png")

    # ── 7. Recommendation Examples ────────────────────────────
    print("\n[7/7] Recommendation Examples for 3 Users")
    print("=" * 80)

    # Pick 3 users with enough test ratings
    example_users = []
    for uid in eval_users:
        user_test_count = len(test[test[config.USER_COL] == uid])
        if user_test_count >= 5:
            example_users.append(uid)
        if len(example_users) >= 3:
            break

    for uid in example_users:
        print(f"\n{'─' * 80}")
        print(f"USER {uid}")
        # Show what they rated in training
        user_train = train[train[config.USER_COL] == uid].merge(
            items[[config.ITEM_COL, config.TITLE_COL]], on=config.ITEM_COL
        )
        top_rated = user_train.nlargest(5, config.RATING_COL)
        print(f"  Top-rated movies (training):")
        for _, row in top_rated.iterrows():
            print(f"    {row[config.RATING_COL]:.1f} - {row[config.TITLE_COL]}")

        # Show recommendations from each model
        for name, model in models.items():
            recs = model.recommend(uid, train, n=5, exclude_seen=True)
            rec_titles = []
            for iid in recs:
                title_match = items.loc[items[config.ITEM_COL] == iid, config.TITLE_COL]
                title = title_match.values[0] if len(title_match) > 0 else f"Item {iid}"
                rec_titles.append(title)
            print(f"\n  {name}:")
            for t in rec_titles:
                print(f"    -> {t}")

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)

    return results_df


if __name__ == "__main__":
    main()
