# CineMatch -- Movie Recommender System

Recommender Systems course project (Prof. Marc Torrens, ESADE 2025).

## Algorithms

All formulas implemented exactly as in lecture slides (verified via formula audit):

| Algorithm | Formula Source | Key Metric |
|---|---|---|
| Most Popular | CollaborativeFiltering.pdf, Folie 12 | Precision@10 = 0.122 |
| Highest Average | CollaborativeFiltering.pdf, Folie 12 | Precision@10 = 0.046 |
| Random | Evaluation.pdf, Folie 28 | Control baseline |
| Content-Based (TF-IDF) | ContentBasedFiltering.pdf, Folien 27-43 | Coverage = 20.6% |
| Content-Based + Tags | ContentBasedFiltering.pdf, Folie 17 | Novelty = 7.12 |
| User-User CF (Pearson) | CollaborativeFiltering.pdf, Folien 15, 18 | RMSE = 0.893 |
| Item-Item CF (Adj. Cosine) | CollaborativeFiltering.pdf, Folien 26, 29 | RMSE = 0.935 |
| Matrix Factorization (SGD) | MatrixFactorization.pdf + Koren (2009) | Precision@10 = 0.053, RMSE = 0.881 |

## Dataset

MovieLens Latest Small (GroupLens): 100,836 ratings, 610 users, 9,742 movies.

Place data files in `data/raw/`:
- `ratings.csv` (userId, movieId, rating, timestamp)
- `movies.csv` (movieId, title, genres)
- `tags.csv` (userId, movieId, tag, timestamp)
- `links.csv` (movieId, imdbId, tmdbId)

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Run full evaluation (all 8 models, all metrics, generates figures)
python run_evaluation.py

# Run the Streamlit prototype
streamlit run app.py

# Build the presentation (requires evaluation results)
python build_presentation.py
```

## Project Structure

```
src/
  config.py               Configuration and paths
  data_loading.py          Data loading and train/test split
  baselines.py             Most Popular, Highest Average, Random
  content_based.py         TF-IDF + cosine similarity
  collaborative_filtering.py  User-User CF (Pearson) + Item-Item CF (Adj. Cosine)
  matrix_factorization.py  Biased SGD (mu + b_u + b_i + p_u * q_i)
  evaluation.py            All metrics (Precision, Recall, NDCG, MRR, Coverage, etc.)
  poster_service.py        TMDb poster URL service with lazy loading + JSON cache

app.py                     Streamlit UI (Netflix-style, 6 pages)
run_evaluation.py          Full evaluation pipeline (8 models, 14 metrics, 8 figures)
build_presentation.py      Generate .pptx from evaluation results
main.py                    Alternative pipeline runner

results/
  metrics.csv              Full metric matrix (8 models x 14 metrics)
  rating_prediction_metrics.csv  MAE/RMSE for applicable models
  scalability.csv           Training times and complexity
  figures/                  8 PNG visualizations

CineMatch_Presentation.pptx  16-slide presentation with embedded figures
```

## Evaluation Metrics

Rating prediction: MAE, RMSE. Top-N: Precision@K, Recall@K, NDCG@K, MRR, Hit Rate@K.
Beyond accuracy: Coverage, Diversity, Novelty, Popularity Bias, Serendipity, Fairness.

## Poster Service

Movie posters are fetched from TMDb API on first request and cached locally.
Set your API key in `.streamlit/secrets.toml`:

```toml
TMDB_API_KEY = "your_key_here"
```
