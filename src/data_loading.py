"""Data loading and preprocessing utilities."""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from . import config


def load_ratings(path=config.RATINGS_PATH):
    """Load user-item ratings.

    Expected MovieLens columns: userId, movieId, rating, timestamp
    """
    df = pd.read_csv(path)
    required = {config.USER_COL, config.ITEM_COL, config.RATING_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return df


def load_items(path=config.ITEMS_PATH):
    """Load item metadata.

    Expected MovieLens columns: movieId, title, genres
    """
    df = pd.read_csv(path)
    required = {config.ITEM_COL, config.TITLE_COL, config.GENRES_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return df


def load_tags(path=config.RAW_DATA_DIR / "tags.csv"):
    """Load user-generated tags."""
    return pd.read_csv(path)


def describe_dataset(ratings, items=None):
    """Print and return basic dataset statistics."""
    n_users = ratings[config.USER_COL].nunique()
    n_items = ratings[config.ITEM_COL].nunique()
    n_ratings = len(ratings)
    sparsity = 1.0 - n_ratings / (n_users * n_items)

    stats = {
        "n_users": n_users,
        "n_items": n_items,
        "n_ratings": n_ratings,
        "sparsity": sparsity,
        "mean_rating": ratings[config.RATING_COL].mean(),
        "median_rating": ratings[config.RATING_COL].median(),
        "min_rating": ratings[config.RATING_COL].min(),
        "max_rating": ratings[config.RATING_COL].max(),
    }

    print("=" * 50)
    print("DATASET STATISTICS")
    print("=" * 50)
    print(f"Users:          {n_users:,}")
    print(f"Items:          {n_items:,}")
    print(f"Ratings:        {n_ratings:,}")
    print(f"Sparsity:       {sparsity:.4%}")
    print(f"Mean rating:    {stats['mean_rating']:.2f}")
    print(f"Median rating:  {stats['median_rating']:.1f}")
    print(f"Rating range:   [{stats['min_rating']}, {stats['max_rating']}]")

    # Rating distribution
    print("\nRATING DISTRIBUTION:")
    dist = ratings[config.RATING_COL].value_counts().sort_index()
    for val, count in dist.items():
        pct = count / n_ratings * 100
        print(f"  {val:>4}: {count:>6,} ({pct:5.1f}%)")

    # Ratings per user
    ratings_per_user = ratings.groupby(config.USER_COL).size()
    print(f"\nRATINGS PER USER:")
    print(f"  Mean:   {ratings_per_user.mean():.1f}")
    print(f"  Median: {ratings_per_user.median():.1f}")
    print(f"  Min:    {ratings_per_user.min()}")
    print(f"  Max:    {ratings_per_user.max()}")

    # Most active users
    top_users = ratings_per_user.nlargest(5)
    print("\nMOST ACTIVE USERS:")
    for uid, cnt in top_users.items():
        print(f"  User {uid}: {cnt} ratings")

    # Most rated items
    ratings_per_item = ratings.groupby(config.ITEM_COL).size()
    if items is not None:
        top_items = ratings_per_item.nlargest(5)
        print("\nMOST POPULAR ITEMS:")
        for iid, cnt in top_items.items():
            title_row = items.loc[items[config.ITEM_COL] == iid, config.TITLE_COL]
            title = title_row.values[0] if len(title_row) > 0 else f"Item {iid}"
            print(f"  {title}: {cnt} ratings")

    print("=" * 50)
    return stats


def train_test_split_ratings(ratings, test_size=0.2, random_state=config.RANDOM_STATE):
    """Create a train/test split (random, stratified by user)."""
    train, test = train_test_split(
        ratings,
        test_size=test_size,
        random_state=random_state,
        stratify=ratings[config.USER_COL],
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def get_seen_items(ratings, user_id):
    """Return the set of items already rated by one user."""
    return set(ratings.loc[ratings[config.USER_COL] == user_id, config.ITEM_COL])
