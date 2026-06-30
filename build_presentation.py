"""Build the final CineMatch presentation (16 slides) with real metrics and figures."""

import os
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

RESULTS = Path("results")
FIGURES = RESULTS / "figures"
OUT = Path("CineMatch_Presentation.pptx")

# Colours
BG      = RGBColor(0x14, 0x14, 0x14)
SURFACE = RGBColor(0x1C, 0x1C, 0x1C)
ACCENT  = RGBColor(0xE8, 0x40, 0x3E)
WHITE   = RGBColor(0xE5, 0xE5, 0xE5)
GREY    = RGBColor(0x80, 0x80, 0x80)
DIM     = RGBColor(0x4A, 0x4A, 0x4A)

W = Inches(13.333)
H = Inches(7.5)


def set_bg(slide, color=BG):
    bg = slide.background
    bg.fill.solid()
    bg.fill.fore_color.rgb = color


def add_text(slide, text, left, top, width, height,
             font_size=18, color=WHITE, bold=False, alignment=PP_ALIGN.LEFT,
             font_name="Calibri"):
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(font_size)
    p.font.color.rgb = color
    p.font.bold = bold
    p.font.name = font_name
    p.alignment = alignment
    return tf


def add_bullet_slide(slide, title, bullets, start_top=Inches(1.8)):
    set_bg(slide)
    add_text(slide, title, Inches(0.8), Inches(0.5), Inches(11), Inches(1),
             font_size=36, bold=True, color=WHITE)
    # Accent bar
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE,
                                    Inches(0.8), Inches(1.3), Inches(1.5), Inches(0.06))
    shape.fill.solid()
    shape.fill.fore_color.rgb = ACCENT
    shape.line.fill.background()

    y = start_top
    for bullet in bullets:
        add_text(slide, bullet, Inches(1.0), y, Inches(10.5), Inches(0.5),
                 font_size=16, color=WHITE)
        y += Inches(0.45)


def add_image_slide(slide, title, img_path, caption=""):
    set_bg(slide)
    add_text(slide, title, Inches(0.8), Inches(0.4), Inches(11), Inches(0.8),
             font_size=32, bold=True, color=WHITE)
    if os.path.exists(img_path):
        slide.shapes.add_picture(str(img_path), Inches(0.8), Inches(1.5),
                                  Inches(11.5), Inches(5.3))
    if caption:
        add_text(slide, caption, Inches(0.8), Inches(6.9), Inches(11), Inches(0.5),
                 font_size=12, color=GREY)


def build():
    prs = Presentation()
    prs.slide_width = W
    prs.slide_height = H
    blank = prs.slide_layouts[6]  # blank layout

    # ── Slide 1: Title ──────────────────────────────────────────
    s = prs.slides.add_slide(blank)
    set_bg(s)
    add_text(s, "CINEMATCH", Inches(1), Inches(2), Inches(11), Inches(1.5),
             font_size=72, bold=True, color=ACCENT)
    add_text(s, "Movie Recommender System", Inches(1), Inches(3.5), Inches(11), Inches(0.8),
             font_size=28, color=WHITE)
    add_text(s, "Recommender Systems Course  |  Prof. Marc Torrens  |  ESADE 2025",
             Inches(1), Inches(4.5), Inches(11), Inches(0.5),
             font_size=16, color=GREY)
    add_text(s, "Jan Philipp Gnau", Inches(1), Inches(5.5), Inches(11), Inches(0.5),
             font_size=18, color=WHITE)

    # ── Slide 2: Agenda ─────────────────────────────────────────
    s = prs.slides.add_slide(blank)
    add_bullet_slide(s, "AGENDA", [
        "1. Problem & Dataset",
        "2. Non-Personalized Baselines",
        "3. Content-Based Filtering (TF-IDF)",
        "4. Collaborative Filtering (User-User & Item-Item)",
        "5. Matrix Factorization (Biased SGD)",
        "6. Evaluation: Accuracy Metrics",
        "7. Evaluation: Beyond Accuracy",
        "8. Trade-off Analysis & Fairness",
        "9. UX & Prototype Demo",
        "10. Conclusions & Lessons Learned",
    ])

    # ── Slide 3: Problem & Dataset ──────────────────────────────
    s = prs.slides.add_slide(blank)
    add_bullet_slide(s, "PROBLEM & DATASET", [
        "Goal: recommend movies a user will enjoy, from a large catalog",
        "Dataset: MovieLens Latest Small (GroupLens)",
        "100,836 ratings  |  610 users  |  9,742 movies",
        "Rating scale: 0.5 -- 5.0 (half-star increments)",
        "Sparsity: 98.3% of the user-item matrix is empty",
        "Train/Test split: 80/20 random",
        "Challenge: extreme sparsity limits collaborative methods",
    ])

    # ── Slide 4: EDA ────────────────────────────────────────────
    s = prs.slides.add_slide(blank)
    add_image_slide(s, "EXPLORATORY DATA ANALYSIS",
                    FIGURES / "eda_distributions.png",
                    "Rating distribution, user activity, and item popularity from MovieLens Latest Small.")

    # ── Slide 5: Non-Personalized Baselines ─────────────────────
    s = prs.slides.add_slide(blank)
    add_bullet_slide(s, "NON-PERSONALIZED BASELINES", [
        "Most Popular: rank by interaction count (# ratings)",
        "Highest Average: S(u,i) = mean of all ratings for item i",
        "  -- Minimum 20 ratings filter to avoid noise (CollaborativeFiltering.pdf, Folie 12)",
        "Random: control baseline (EvaluationRecommenderSystems.pdf, Folie 28)",
        "",
        "Results: Most Popular achieves Precision@10 = 0.122 (best overall!)",
        "  -- Crowd favorites are a strong signal in sparse data",
        "  -- But: 100% popularity bias, 0% serendipity, coverage < 1%",
    ])

    # ── Slide 6: Content-Based Filtering ────────────────────────
    s = prs.slides.add_slide(blank)
    add_bullet_slide(s, "CONTENT-BASED FILTERING", [
        "Item representation: TF-IDF on genres (ContentBasedFiltering.pdf, Folie 37)",
        "  -- Optionally enriched with user-generated tags (Folie 17)",
        "User profile: profile(u) = Sigma_i (r_ui - r_u) * vector(i), normalized",
        "  -- (ContentBasedFiltering.pdf, Folien 27-28)",
        "Prediction: score(u,i) = cos(profile_u, vector_i) (Folie 33)",
        "",
        "Results: Precision@10 = 0.009  |  Coverage = 20.6%  |  Diversity = 0.09",
        "Low accuracy but best genre-coherent lists (low diversity = focused)",
        "Very low popularity bias (18.6%) -- good for discovery",
    ])

    # ── Slide 7: Collaborative Filtering ────────────────────────
    s = prs.slides.add_slide(blank)
    add_bullet_slide(s, "COLLABORATIVE FILTERING", [
        "User-User CF (CollaborativeFiltering.pdf, Folie 15):",
        "  S(u,i) = r_u + Sigma_v (r_vi - r_v) * w_uv / Sigma |w_uv|",
        "  Similarity: Pearson Correlation (Folie 18)",
        "",
        "Item-Item CF (CollaborativeFiltering.pdf, Folie 29):",
        "  S(u,i) = r_i + Sigma_j (r_uj - r_j) * w_ij / Sigma |w_ij|",
        "  Similarity: Adjusted Cosine (Folie 26)",
        "",
        "Both near-zero Precision@10 due to 98.3% sparsity",
        "User-User: MAE=0.678, RMSE=0.893 (rating prediction is better)",
        "But: highest novelty (9.2) and zero popularity bias!",
    ])

    # ── Slide 8: Matrix Factorization ───────────────────────────
    s = prs.slides.add_slide(blank)
    add_bullet_slide(s, "MATRIX FACTORIZATION (Biased SGD)", [
        "MatrixFactorization.pdf, Folien 6-7 + Koren et al. (2009):",
        "  r_hat = mu + b_u + b_i + p_u * q_i",
        "  50 latent factors, 20 epochs, lr=0.005, reg=0.02",
        "",
        "SGD: b_u += alpha*(e - lambda*b_u), p_u += alpha*(e*q_i - lambda*p_u)",
        "  Correct p_u_old for q_i update (simultaneous gradient)",
        "",
        "Results: Precision@10 = 0.053  |  RMSE = 0.881  |  MAE = 0.675",
        "Best personalized method for top-N recommendation",
        "Training converges: RMSE 0.86 -> 0.73 over 20 epochs",
    ])

    # ── Slide 9: MF Training Curve ──────────────────────────────
    s = prs.slides.add_slide(blank)
    add_image_slide(s, "MF TRAINING CONVERGENCE",
                    FIGURES / "mf_training_curve.png",
                    "Training RMSE decreases steadily over 20 epochs, confirming proper SGD convergence.")

    # ── Slide 10: Accuracy Metrics ──────────────────────────────
    s = prs.slides.add_slide(blank)
    add_image_slide(s, "ACCURACY: MODEL COMPARISON",
                    FIGURES / "model_comparison.png",
                    "Precision@K, Recall@K, NDCG@K, MRR, Hit Rate across all 8 models.")

    # ── Slide 11: Beyond Accuracy ───────────────────────────────
    s = prs.slides.add_slide(blank)
    add_image_slide(s, "BEYOND ACCURACY",
                    FIGURES / "per_user_distributions.png",
                    "Per-user metric distributions showing variance across users (Precision@K and NDCG@K).")

    # ── Slide 12: Trade-off Analysis ────────────────────────────
    s = prs.slides.add_slide(blank)
    add_image_slide(s, "TRADE-OFF: PRECISION vs DIVERSITY vs NOVELTY",
                    FIGURES / "tradeoff_analysis.png",
                    "Core trade-off from lectures: accuracy vs. diversity/novelty/coverage. No single model dominates all dimensions.")

    # ── Slide 13: Radar Chart ───────────────────────────────────
    s = prs.slides.add_slide(blank)
    add_image_slide(s, "MULTI-DIMENSIONAL MODEL COMPARISON",
                    FIGURES / "radar_chart.png",
                    "Normalized radar chart: each model excels in different dimensions. Hybrid approaches are needed.")

    # ── Slide 14: Fairness & Popularity Bias ────────────────────
    s = prs.slides.add_slide(blank)
    add_bullet_slide(s, "FAIRNESS & POPULARITY BIAS", [
        "Fairness: NDCG@K gap between heavy raters (>50th percentile) and light raters",
        "",
        "Largest gap: Most Popular (0.156) -- heavy raters benefit disproportionately",
        "Matrix Factorization gap: 0.143 -- also biased toward active users",
        "CB+Tags gap: 0.005 -- most equitable across user groups",
        "Random gap: 0.003 -- naturally fair (no personalization)",
        "",
        "Popularity bias: Most Popular = 100% from top 10% items",
        "CF methods = 0% popularity bias (but also near-zero precision)",
        "Production systems need explicit debiasing strategies",
    ])

    # ── Slide 15: UX & Demo ─────────────────────────────────────
    s = prs.slides.add_slide(blank)
    add_bullet_slide(s, "UX & PROTOTYPE", [
        "Netflix-inspired Streamlit UI with 6 pages",
        "Design: dark theme (#141414), Bebas Neue / Inter / JetBrains Mono",
        "Poster lazy loading via TMDb API with JSON cache",
        "",
        "Pages: Home (hero + trending), Discover (search + filter + similar),",
        "  Recommendations (7 algorithms), User Profile (stats + genre radar),",
        "  Compare (side-by-side with overlap heatmap), Evaluation (full metrics)",
        "",
        "Poster cards with hover overlay showing title, year, genres, and 'why'",
        "Horizontal scroll rows mimicking Netflix filmstrip UX",
    ])

    # ── Slide 16: Conclusions ───────────────────────────────────
    s = prs.slides.add_slide(blank)
    add_bullet_slide(s, "CONCLUSIONS & LESSONS LEARNED", [
        "1. All formulas implemented exactly as in lecture slides (verified audit)",
        "2. Most Popular is surprisingly strong in sparse data (Precision@10 = 0.122)",
        "3. CF struggles at 98.3% sparsity -- good discussion material",
        "4. MF is the best personalized method (Precision = 0.053, RMSE = 0.881)",
        "5. Core trade-off: accuracy vs discovery (as lectures emphasize)",
        "6. No single model dominates all dimensions",
        "",
        "Key insight from Netflix Prize (Folie 22):",
        "  'Ensembles win benchmarks' and 'Evaluation defines success'",
        "  RMSE alone does not capture recommendation quality",
    ])

    prs.save(str(OUT))
    print(f"Saved {OUT} ({len(prs.slides)} slides)")


if __name__ == "__main__":
    build()
