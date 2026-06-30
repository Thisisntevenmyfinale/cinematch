"""Comprehensive evaluation analysis for the recommender system.

Generates:
- Detailed metrics comparison table
- Beyond-accuracy analysis (diversity, novelty, bias, coverage)
- Scalability analysis (training time & memory)
- Accuracy vs diversity trade-off scatter plot
- Rating prediction metrics (MAE, RMSE) for applicable models
- Per-user metric distributions
- Recommendation overlap analysis

Saves all results and figures to results/
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
    train_test_split_ratings, get_seen_items,
)
from src.baselines import (
    MostPopularRecommender, HighestAverageRatingRecommender, RandomRecommender,
)
from src.content_based import ContentBasedRecommender
from src.collaborative_filtering import ItemItemCollaborativeFiltering
from src.matrix_factorization import MatrixFactorizationRecommender
from src.evaluation import (
    evaluate_model, precision_at_k, recall_at_k, ndcg_at_k,
    mean_reciprocal_rank, catalog_coverage, intra_list_diversity,
    novelty_score, popularity_bias, serendipity_score, mae, rmse,
)

warnings.filterwarnings("ignore")


def main():
    print("=" * 70)
    print("  COMPREHENSIVE EVALUATION ANALYSIS")
    print("=" * 70)

    # ── Load & Split ─────────────────────────────────────────────
    ratings = load_ratings()
    items = load_items()
    tags = load_tags()
    train, test = train_test_split_ratings(ratings, test_size=0.2)

    item_popularity = train.groupby(config.ITEM_COL).size().to_dict()
    n_users = train[config.USER_COL].nunique()
    sorted_by_pop = sorted(item_popularity.keys(), key=lambda x: item_popularity[x], reverse=True)
    n_top = max(1, int(len(sorted_by_pop) * 0.1))
    popular_items = set(sorted_by_pop[:n_top])

    test_relevant = test[test[config.RATING_COL] >= 4.0]
    eval_users = test_relevant[config.USER_COL].unique()

    # ── Train Models ─────────────────────────────────────────────
    print("\nTraining models...")
    models = {}
    timings = {}

    model_configs = [
        ("Random", RandomRecommender(random_state=42)),
        ("Most Popular", MostPopularRecommender()),
        ("Highest Average", HighestAverageRatingRecommender(min_ratings=20)),
        ("Content-Based", ContentBasedRecommender(use_tags=False)),
        ("CB + Tags", ContentBasedRecommender(use_tags=True)),
        ("Item-Item CF", ItemItemCollaborativeFiltering(k=30)),
        ("Matrix Fact.", MatrixFactorizationRecommender(
            n_factors=50, n_epochs=20, lr=0.005, reg=0.02, verbose=False)),
    ]

    for name, model in model_configs:
        t0 = time.time()
        if name == "CB + Tags":
            model.fit(train, items, tags=tags)
        elif name in ("Content-Based",):
            model.fit(train, items)
        else:
            model.fit(train)
        timings[name] = time.time() - t0
        models[name] = model
        print(f"  {name}: {timings[name]:.2f}s")

    # Get CB features for diversity calculation
    cb_features = models["Content-Based"].item_features_
    cb_id_to_idx = models["Content-Based"].item_id_to_index_

    K = config.TOP_K

    # ── 1. Standard Evaluation ───────────────────────────────────
    print("\n[1] Standard Top-N Evaluation...")
    results = {}
    for name, model in models.items():
        print(f"  {name}...")
        res = evaluate_model(
            model, train, test, eval_users, items_df=items, k=K,
            item_features=cb_features, item_id_to_index=cb_id_to_idx,
            item_popularity=item_popularity, n_users=n_users,
            popular_items=popular_items,
        )
        res["Training Time (s)"] = timings[name]
        results[name] = res

    results_df = pd.DataFrame(results).T
    results_df.index.name = "Model"
    results_df.to_csv(config.RESULTS_DIR / "metrics.csv")

    print("\nMetrics Table:")
    print(results_df.to_string(float_format=lambda x: f"{x:.4f}"))

    # ── 2. Rating Prediction Metrics (MAE/RMSE) ────────────────
    print("\n[2] Rating Prediction Metrics (MAE / RMSE)...")
    pred_models = {
        "Item-Item CF": models["Item-Item CF"],
        "Matrix Fact.": models["Matrix Fact."],
    }

    pred_results = {}
    for name, model in pred_models.items():
        predictions = []
        actuals = []
        sample_test = test.sample(n=min(2000, len(test)), random_state=42)
        for _, row in sample_test.iterrows():
            uid = row[config.USER_COL]
            iid = row[config.ITEM_COL]
            try:
                pred = model.predict_score(uid, iid)
                predictions.append(pred)
                actuals.append(row[config.RATING_COL])
            except Exception:
                pass
        if predictions:
            pred_results[name] = {
                "MAE": mae(predictions, actuals),
                "RMSE": rmse(predictions, actuals),
                "N_predictions": len(predictions),
            }
            print(f"  {name}: MAE={pred_results[name]['MAE']:.4f}, RMSE={pred_results[name]['RMSE']:.4f}")

    if pred_results:
        pred_df = pd.DataFrame(pred_results).T
        pred_df.to_csv(config.RESULTS_DIR / "rating_prediction_metrics.csv")

    # ── 3. Per-User Metric Distributions ────────────────────────
    print("\n[3] Per-User Metric Distributions...")
    per_user_data = {}
    sample_users = eval_users[:100]  # sample for visualization

    for name, model in [("Most Popular", models["Most Popular"]),
                         ("Content-Based", models["Content-Based"]),
                         ("Matrix Fact.", models["Matrix Fact."])]:
        user_precisions = []
        user_ndcgs = []
        for uid in sample_users:
            user_test = test[test[config.USER_COL] == uid]
            relevant = set(user_test.loc[user_test[config.RATING_COL] >= 4.0, config.ITEM_COL])
            if not relevant:
                continue
            recs = model.recommend(uid, train, n=K, exclude_seen=True)
            user_precisions.append(precision_at_k(recs, relevant, K))
            user_ndcgs.append(ndcg_at_k(recs, relevant, K))
        per_user_data[name] = {"precision": user_precisions, "ndcg": user_ndcgs}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for name, data in per_user_data.items():
        axes[0].hist(data["precision"], bins=20, alpha=0.5, label=name)
        axes[1].hist(data["ndcg"], bins=20, alpha=0.5, label=name)
    axes[0].set_title(f"Precision@{K} Distribution (per user)")
    axes[0].set_xlabel(f"Precision@{K}")
    axes[0].legend()
    axes[1].set_title(f"NDCG@{K} Distribution (per user)")
    axes[1].set_xlabel(f"NDCG@{K}")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(config.RESULTS_DIR / "figures" / "per_user_distributions.png", dpi=150)
    plt.close()
    print("  Saved per-user distributions")

    # ── 4. Accuracy vs Beyond-Accuracy Trade-offs ───────────────
    print("\n[4] Trade-off Analysis...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 4a: Precision vs Diversity
    ax = axes[0]
    for model_name in results_df.index:
        if "Diversity" in results_df.columns:
            ax.scatter(
                results_df.loc[model_name, "Precision@K"],
                results_df.loc[model_name, "Diversity"],
                s=120, zorder=5,
            )
            ax.annotate(model_name,
                        (results_df.loc[model_name, "Precision@K"],
                         results_df.loc[model_name, "Diversity"]),
                        textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xlabel("Precision@K")
    ax.set_ylabel("Diversity")
    ax.set_title("Precision vs Diversity")

    # 4b: Precision vs Novelty
    ax = axes[1]
    for model_name in results_df.index:
        if "Novelty" in results_df.columns:
            ax.scatter(
                results_df.loc[model_name, "Precision@K"],
                results_df.loc[model_name, "Novelty"],
                s=120, zorder=5,
            )
            ax.annotate(model_name,
                        (results_df.loc[model_name, "Precision@K"],
                         results_df.loc[model_name, "Novelty"]),
                        textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xlabel("Precision@K")
    ax.set_ylabel("Novelty")
    ax.set_title("Precision vs Novelty")

    # 4c: Coverage vs Popularity Bias
    ax = axes[2]
    for model_name in results_df.index:
        if "Popularity Bias" in results_df.columns:
            ax.scatter(
                results_df.loc[model_name, "Coverage"],
                results_df.loc[model_name, "Popularity Bias"],
                s=120, zorder=5,
            )
            ax.annotate(model_name,
                        (results_df.loc[model_name, "Coverage"],
                         results_df.loc[model_name, "Popularity Bias"]),
                        textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Popularity Bias")
    ax.set_title("Coverage vs Popularity Bias")

    plt.tight_layout()
    plt.savefig(config.RESULTS_DIR / "figures" / "tradeoff_analysis.png", dpi=150)
    plt.close()
    print("  Saved trade-off analysis plots")

    # ── 5. Recommendation Overlap Analysis ──────────────────────
    print("\n[5] Recommendation Overlap Analysis...")
    model_names = list(models.keys())
    overlap_matrix = np.zeros((len(model_names), len(model_names)))

    sample_overlap_users = eval_users[:50]
    for uid in sample_overlap_users:
        recs_per_model = {}
        for name, model in models.items():
            recs_per_model[name] = set(model.recommend(uid, train, n=K, exclude_seen=True))

        for i, m1 in enumerate(model_names):
            for j, m2 in enumerate(model_names):
                if recs_per_model[m1] and recs_per_model[m2]:
                    overlap = len(recs_per_model[m1] & recs_per_model[m2]) / K
                    overlap_matrix[i, j] += overlap

    overlap_matrix /= len(sample_overlap_users)

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(overlap_matrix, cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_names, fontsize=8)
    for i in range(len(model_names)):
        for j in range(len(model_names)):
            ax.text(j, i, f"{overlap_matrix[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title(f"Recommendation Overlap (avg Jaccard over {len(sample_overlap_users)} users)")
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(config.RESULTS_DIR / "figures" / "recommendation_overlap.png", dpi=150)
    plt.close()
    print("  Saved overlap heatmap")

    # ── 6. Popularity Bias Deep Dive ────────────────────────────
    print("\n[6] Popularity Bias Analysis...")
    fig, ax = plt.subplots(figsize=(10, 5))

    item_pop_series = train.groupby(config.ITEM_COL).size()
    for name, model in [("Most Popular", models["Most Popular"]),
                         ("Content-Based", models["Content-Based"]),
                         ("Matrix Fact.", models["Matrix Fact."]),
                         ("Random", models["Random"])]:
        rec_items = []
        for uid in sample_overlap_users:
            rec_items.extend(model.recommend(uid, train, n=K, exclude_seen=True))

        rec_pops = [item_pop_series.get(i, 0) for i in rec_items]
        ax.hist(rec_pops, bins=30, alpha=0.5, label=name, density=True)

    ax.set_xlabel("Item Popularity (# ratings)")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of Recommended Items by Popularity")
    ax.legend()
    plt.tight_layout()
    plt.savefig(config.RESULTS_DIR / "figures" / "popularity_bias_distribution.png", dpi=150)
    plt.close()
    print("  Saved popularity bias distribution")

    # ── 7. Scalability Summary ──────────────────────────────────
    print("\n[7] Scalability Summary")
    scalability = pd.DataFrame({
        "Model": list(timings.keys()),
        "Training Time (s)": list(timings.values()),
        "Complexity": [
            "O(1)",                          # Random
            "O(n_ratings)",                  # Most Popular
            "O(n_ratings)",                  # Highest Average
            "O(n_items * n_features)",       # Content-Based
            "O(n_items * n_features)",       # CB + Tags
            "O(n_users * n_items²)",         # Item-Item CF
            "O(n_epochs * n_ratings * k)",   # MF
        ],
    })
    scalability.to_csv(config.RESULTS_DIR / "scalability.csv", index=False)
    print(scalability.to_string(index=False))

    # ── 8. MF Training Curve ────────────────────────────────────
    print("\n[8] Matrix Factorization Training Curve...")
    mf_model = models["Matrix Fact."]
    if hasattr(mf_model, "training_losses_") and mf_model.training_losses_:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(range(1, len(mf_model.training_losses_) + 1),
                mf_model.training_losses_, marker="o")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("RMSE")
        ax.set_title("Matrix Factorization: Training RMSE over Epochs")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(config.RESULTS_DIR / "figures" / "mf_training_curve.png", dpi=150)
        plt.close()
        print("  Saved MF training curve")

    # ── Summary ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  EVALUATION COMPLETE")
    print("=" * 70)
    print(f"\nFiles saved to {config.RESULTS_DIR}/:")
    print("  - metrics.csv")
    print("  - rating_prediction_metrics.csv")
    print("  - scalability.csv")
    print("  - figures/eda_distributions.png")
    print("  - figures/model_comparison.png")
    print("  - figures/per_user_distributions.png")
    print("  - figures/tradeoff_analysis.png")
    print("  - figures/recommendation_overlap.png")
    print("  - figures/popularity_bias_distribution.png")
    print("  - figures/mf_training_curve.png")


if __name__ == "__main__":
    main()
