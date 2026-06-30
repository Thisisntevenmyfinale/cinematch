"""Matrix Factorization recommender using SGD (Stochastic Gradient Descent).

Implements biased matrix factorization as per lecture slides and Netflix Prize:

    r̂_ui = μ + b_u + b_i + p_u · q_i

where:
    μ   = global mean rating
    b_u = user bias
    b_i = item bias
    p_u = user latent factor vector (1 x n_factors)
    q_i = item latent factor vector (1 x n_factors)

Objective function (minimise):
    Σ_(u,i)∈known (r_ui - r̂_ui)² + λ(||p_u||² + ||q_i||² + b_u² + b_i²)

SGD updates:
    e_ui = r_ui - r̂_ui
    b_u ← b_u + α(e_ui - λ·b_u)
    b_i ← b_i + α(e_ui - λ·b_i)
    p_u ← p_u + α(e_ui·q_i - λ·p_u)
    q_i ← q_i + α(e_ui·p_u - λ·q_i)
"""

import numpy as np
import pandas as pd
from . import config


class MatrixFactorizationRecommender:
    """Biased Matrix Factorization with SGD.

    r̂_ui = μ + b_u + b_i + p_u · q_i
    """

    def __init__(self, n_factors=50, n_epochs=20, lr=0.005, reg=0.02,
                 random_state=42, verbose=True):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.random_state = random_state
        self.verbose = verbose

        self.global_mean_ = None
        self.user_bias_ = None
        self.item_bias_ = None
        self.P_ = None  # user factors (n_users x n_factors)
        self.Q_ = None  # item factors (n_items x n_factors)
        self.user_id_to_index_ = None
        self.item_id_to_index_ = None
        self.user_ids_ = None
        self.item_ids_ = None
        self.all_items_ = None
        self.training_losses_ = []

    def fit(self, ratings):
        """Train biased MF model using SGD on known ratings."""
        rng = np.random.RandomState(self.random_state)

        self.user_ids_ = np.sort(ratings[config.USER_COL].unique())
        self.item_ids_ = np.sort(ratings[config.ITEM_COL].unique())
        self.all_items_ = set(self.item_ids_)
        self.user_id_to_index_ = {
            uid: idx for idx, uid in enumerate(self.user_ids_)
        }
        self.item_id_to_index_ = {
            iid: idx for idx, iid in enumerate(self.item_ids_)
        }

        n_users = len(self.user_ids_)
        n_items = len(self.item_ids_)

        self.global_mean_ = ratings[config.RATING_COL].mean()
        self.user_bias_ = np.zeros(n_users)
        self.item_bias_ = np.zeros(n_items)
        self.P_ = rng.normal(0, 0.1, (n_users, self.n_factors))
        self.Q_ = rng.normal(0, 0.1, (n_items, self.n_factors))

        # Convert to arrays for fast iteration
        users = ratings[config.USER_COL].map(self.user_id_to_index_).values
        items = ratings[config.ITEM_COL].map(self.item_id_to_index_).values
        vals = ratings[config.RATING_COL].values.astype(np.float64)

        self.training_losses_ = []

        for epoch in range(self.n_epochs):
            # Shuffle training data each epoch
            indices = rng.permutation(len(vals))
            total_loss = 0.0

            for idx in indices:
                u = users[idx]
                i = items[idx]
                r = vals[idx]

                # Predict: r̂_ui = μ + b_u + b_i + p_u · q_i
                pred = (
                    self.global_mean_
                    + self.user_bias_[u]
                    + self.item_bias_[i]
                    + self.P_[u] @ self.Q_[i]
                )

                # Error
                e = r - pred
                total_loss += e ** 2

                # SGD updates
                self.user_bias_[u] += self.lr * (e - self.reg * self.user_bias_[u])
                self.item_bias_[i] += self.lr * (e - self.reg * self.item_bias_[i])

                p_u_old = self.P_[u].copy()
                self.P_[u] += self.lr * (e * self.Q_[i] - self.reg * self.P_[u])
                self.Q_[i] += self.lr * (e * p_u_old - self.reg * self.Q_[i])

            rmse = np.sqrt(total_loss / len(vals))
            self.training_losses_.append(rmse)
            if self.verbose and (epoch + 1) % 5 == 0:
                print(f"  MF Epoch {epoch + 1}/{self.n_epochs}: RMSE = {rmse:.4f}")

        return self

    def predict_score(self, user_id, item_id):
        """Predict score: r̂_ui = μ + b_u + b_i + p_u · q_i"""
        u_known = user_id in self.user_id_to_index_
        i_known = item_id in self.item_id_to_index_

        pred = self.global_mean_

        if u_known:
            u_idx = self.user_id_to_index_[user_id]
            pred += self.user_bias_[u_idx]
        if i_known:
            i_idx = self.item_id_to_index_[item_id]
            pred += self.item_bias_[i_idx]
        if u_known and i_known:
            pred += self.P_[u_idx] @ self.Q_[i_idx]

        return pred

    def recommend(self, user_id, ratings_train, n=10, exclude_seen=True):
        """Recommend top-n items by predicted score."""
        if user_id not in self.user_id_to_index_:
            return []

        u_idx = self.user_id_to_index_[user_id]

        # Vectorised scoring: μ + b_u + b_i + P[u] · Q^T
        scores = (
            self.global_mean_
            + self.user_bias_[u_idx]
            + self.item_bias_
            + self.Q_ @ self.P_[u_idx]
        )

        if exclude_seen:
            seen = set(
                ratings_train.loc[
                    ratings_train[config.USER_COL] == user_id, config.ITEM_COL
                ]
            )
            seen_indices = [self.item_id_to_index_[i] for i in seen if i in self.item_id_to_index_]
            scores[seen_indices] = -np.inf

        top_indices = np.argsort(scores)[::-1][:n]
        return [self.item_ids_[i] for i in top_indices]
