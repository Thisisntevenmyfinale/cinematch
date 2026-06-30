"""CineMatch — Movie Recommender System Prototype.

Interactive UI demonstrating Non-Personalized, Content-Based,
Collaborative Filtering, and Matrix Factorization on MovieLens Latest Small.

Run: streamlit run app.py
"""

import re
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

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

# ─────────────────────────────────────────────────────────────────
# DESIGN SYSTEM
# ─────────────────────────────────────────────────────────────────
# Palette inspired by cinema: dimming lights, projector warmth, curtain fabric
COLORS = {
    "saal":      "#14121a",   # deep violet-black background
    "vorhang":   "#1e1b2e",   # surface / cards
    "vorhang_l": "#282440",   # lighter surface for hover / active
    "projektor": "#c9a55a",   # warm gold accent
    "ticket":    "#d64550",   # muted red accent
    "leinwand":  "#e8e4dc",   # warm white text
    "schatten":  "#7a7486",   # muted lavender-grey
    "perf":      "#3a3550",   # filmstrip perforation dots
}

# Matplotlib theme matching the palette
plt.rcParams.update({
    "figure.facecolor": COLORS["saal"],
    "axes.facecolor":   COLORS["vorhang"],
    "axes.edgecolor":   COLORS["schatten"],
    "axes.labelcolor":  COLORS["leinwand"],
    "text.color":       COLORS["leinwand"],
    "xtick.color":      COLORS["schatten"],
    "ytick.color":      COLORS["schatten"],
    "grid.color":       COLORS["perf"],
    "grid.alpha":       0.3,
    "legend.facecolor": COLORS["vorhang"],
    "legend.edgecolor": COLORS["perf"],
    "font.family":      "sans-serif",
})
CHART_COLORS = [
    COLORS["projektor"], COLORS["ticket"], "#5b8fa8",
    "#8b6fb0", "#6aaa64", COLORS["schatten"], "#c07040",
]


# ─────────────────────────────────────────────────────────────────
# PAGE CONFIG & GLOBAL CSS
# ─────────────────────────────────────────────────────────────────
st.set_page_config(page_title="CineMatch", page_icon="C", layout="wide")

st.markdown(f"""
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@500;700&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
    :root {{
        --saal:      {COLORS['saal']};
        --vorhang:   {COLORS['vorhang']};
        --vorhang-l: {COLORS['vorhang_l']};
        --projektor: {COLORS['projektor']};
        --ticket:    {COLORS['ticket']};
        --leinwand:  {COLORS['leinwand']};
        --schatten:  {COLORS['schatten']};
        --perf:      {COLORS['perf']};
    }}

    /* ── Global overrides ─────────────────────────────── */
    .stApp {{
        background-color: var(--saal);
        color: var(--leinwand);
        font-family: 'Inter', sans-serif;
    }}
    #MainMenu, footer, header {{visibility: hidden;}}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background-color: var(--vorhang);
        border-right: 1px solid var(--perf);
    }}
    section[data-testid="stSidebar"] .stRadio label {{
        font-family: 'Inter', sans-serif;
        font-weight: 400;
        font-size: 0.92rem;
        color: var(--schatten);
        padding: 6px 0;
    }}
    section[data-testid="stSidebar"] .stRadio label[data-checked="true"],
    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:has(input:checked) {{
        color: var(--projektor) !important;
        font-weight: 500;
    }}

    /* Headings */
    h1, h2, h3 {{
        font-family: 'Playfair Display', serif !important;
        color: var(--leinwand) !important;
        font-weight: 700;
    }}
    h1 {{ letter-spacing: -0.5px; }}

    /* Links */
    a {{ color: var(--projektor); }}

    /* Buttons */
    .stButton > button {{
        background-color: var(--projektor) !important;
        color: var(--saal) !important;
        border: none !important;
        border-radius: 6px !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        letter-spacing: 0.3px;
        transition: opacity 0.15s;
    }}
    .stButton > button:hover {{
        opacity: 0.85;
    }}

    /* Dataframes */
    .stDataFrame {{ border-radius: 8px; overflow: hidden; }}

    /* Tabs */
    .stTabs [data-baseweb="tab"] {{
        color: var(--schatten);
        font-family: 'Inter', sans-serif;
        font-weight: 500;
    }}
    .stTabs [data-baseweb="tab"][aria-selected="true"] {{
        color: var(--projektor);
        border-bottom-color: var(--projektor);
    }}

    /* Inputs / selectboxes */
    .stSelectbox label, .stTextInput label, .stSlider label,
    .stNumberInput label, .stMultiSelect label {{
        color: var(--schatten) !important;
        font-family: 'Inter', sans-serif;
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}

    /* Expander */
    .streamlit-expanderHeader {{
        color: var(--schatten) !important;
        font-family: 'Inter', sans-serif;
    }}

    /* Metric overrides */
    [data-testid="stMetric"] {{
        background: var(--vorhang);
        border-radius: 8px;
        padding: 12px 16px;
        border: 1px solid var(--perf);
    }}
    [data-testid="stMetricValue"] {{
        font-family: 'JetBrains Mono', monospace !important;
        color: var(--projektor) !important;
    }}
    [data-testid="stMetricLabel"] {{
        color: var(--schatten) !important;
        font-family: 'Inter', sans-serif;
        text-transform: uppercase;
        font-size: 0.72rem !important;
        letter-spacing: 0.8px;
    }}

    /* ── Filmstrip Card (signature element) ───────────── */
    .fs-card {{
        display: flex;
        background: var(--vorhang);
        border-radius: 0 10px 10px 0;
        margin-bottom: 10px;
        overflow: hidden;
        transition: background 0.15s;
    }}
    .fs-card:hover {{
        background: var(--vorhang-l);
    }}
    .fs-strip {{
        width: 6px;
        flex-shrink: 0;
        background:
            repeating-linear-gradient(
                to bottom,
                var(--perf) 0px,
                var(--perf) 4px,
                transparent 4px,
                transparent 10px
            ),
            linear-gradient(to bottom, var(--ticket), var(--projektor));
        background-blend-mode: overlay;
    }}
    .fs-body {{
        padding: 14px 18px;
        flex-grow: 1;
        min-width: 0;
    }}
    .fs-rank {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        color: var(--projektor);
        letter-spacing: 1px;
        margin-bottom: 2px;
    }}
    .fs-title {{
        font-family: 'Playfair Display', serif;
        font-size: 1.05rem;
        color: var(--leinwand);
        font-weight: 500;
        line-height: 1.3;
    }}
    .fs-title .fs-year {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        color: var(--schatten);
        font-weight: 400;
        margin-left: 6px;
    }}
    .fs-genres {{
        margin-top: 5px;
    }}
    .fs-genres span {{
        display: inline-block;
        font-family: 'Inter', sans-serif;
        font-size: 0.68rem;
        font-weight: 500;
        color: var(--schatten);
        background: var(--perf);
        padding: 1px 8px;
        border-radius: 3px;
        margin-right: 4px;
        margin-bottom: 3px;
        letter-spacing: 0.3px;
    }}
    .fs-why {{
        font-family: 'Inter', sans-serif;
        font-size: 0.78rem;
        color: var(--schatten);
        margin-top: 6px;
        font-style: italic;
    }}
    .fs-stat {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        color: var(--schatten);
        margin-top: 5px;
    }}
    .fs-stat em {{
        color: var(--projektor);
        font-style: normal;
    }}

    /* ── Algorithm property card ──────────────────────── */
    .algo-card {{
        background: var(--vorhang);
        border-radius: 8px;
        padding: 16px 18px;
        margin-bottom: 10px;
        border: 1px solid var(--perf);
    }}
    .algo-card h4 {{
        font-family: 'Inter', sans-serif !important;
        font-weight: 600;
        font-size: 0.92rem;
        color: var(--leinwand) !important;
        margin: 0 0 6px 0;
    }}
    .algo-card .algo-desc {{
        font-size: 0.82rem;
        color: var(--schatten);
        line-height: 1.5;
        margin-bottom: 10px;
    }}
    .algo-card .algo-props {{
        display: flex;
        gap: 16px;
    }}
    .algo-card .algo-prop {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        color: var(--schatten);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    .algo-card .algo-prop em {{
        font-style: normal;
        color: var(--projektor);
    }}
    .algo-bar {{
        display: inline-block;
        height: 4px;
        border-radius: 2px;
        vertical-align: middle;
        margin-left: 4px;
    }}

    /* ── Intro prose highlight numbers ────────────────── */
    .prose-num {{
        font-family: 'JetBrains Mono', monospace;
        color: var(--projektor);
        font-weight: 500;
    }}

    /* ── Section divider ─────────────────────────────── */
    .sec-div {{
        border: none;
        border-top: 1px solid var(--perf);
        margin: 28px 0 24px 0;
    }}

    /* ── Explanation callout ──────────────────────────── */
    .callout {{
        background: var(--vorhang);
        border-left: 3px solid var(--projektor);
        padding: 12px 16px;
        border-radius: 0 6px 6px 0;
        margin: 12px 0;
        font-size: 0.85rem;
        color: var(--schatten);
        line-height: 1.6;
    }}
    .callout strong {{ color: var(--leinwand); }}

    /* ── Sidebar brand ───────────────────────────────── */
    .sidebar-brand {{
        font-family: 'Playfair Display', serif;
        font-size: 1.5rem;
        color: var(--leinwand);
        font-weight: 700;
        letter-spacing: -0.5px;
        padding: 0 0 4px 0;
    }}
    .sidebar-brand-sub {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.65rem;
        color: var(--schatten);
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────
def extract_year(title):
    m = re.search(r"\((\d{4})\)", str(title))
    return m.group(1) if m else ""


def genre_pills_html(genres_str):
    if not genres_str or genres_str == "(no genres listed)":
        return ""
    return "".join(
        f"<span>{g.strip()}</span>" for g in str(genres_str).split("|")
        if g.strip() and g.strip() != "(no genres listed)"
    )


def filmstrip_card(rank, title, genres_str, why="", stat_line=""):
    """Render the signature filmstrip movie card."""
    year = extract_year(title)
    clean = re.sub(r"\s*\(\d{4}\)\s*$", "", str(title))
    rank_label = f"#{rank:02d}" if isinstance(rank, int) else str(rank)
    why_html = f'<div class="fs-why">{why}</div>' if why else ""
    stat_html = f'<div class="fs-stat">{stat_line}</div>' if stat_line else ""
    return f"""<div class="fs-card">
        <div class="fs-strip"></div>
        <div class="fs-body">
            <div class="fs-rank">{rank_label}</div>
            <div class="fs-title">{clean}<span class="fs-year">{year}</span></div>
            <div class="fs-genres">{genre_pills_html(genres_str)}</div>
            {why_html}{stat_html}
        </div>
    </div>"""


def rating_label(rating):
    """Format a rating as a short label."""
    return f"{rating:.1f}/5"


def algo_prop_bar(value, max_val=3, color=None):
    """Small inline bar for algorithm property visualization."""
    c = color or COLORS["projektor"]
    filled_w = int(value / max_val * 36)
    empty_w = 36 - filled_w
    return (f'<span class="algo-bar" style="width:{filled_w}px; background:{c};"></span>'
            f'<span class="algo-bar" style="width:{empty_w}px; background:var(--perf);"></span>')


def get_user_genre_profile(user_ratings, items_df):
    merged = user_ratings.merge(items_df[[config.ITEM_COL, config.GENRES_COL]], on=config.ITEM_COL)
    genre_scores = {}
    genre_counts = {}
    for _, row in merged.iterrows():
        for g in str(row[config.GENRES_COL]).split("|"):
            g = g.strip()
            if g and g != "(no genres listed)":
                genre_scores[g] = genre_scores.get(g, 0) + row[config.RATING_COL]
                genre_counts[g] = genre_counts.get(g, 0) + 1
    return {g: genre_scores[g] / genre_counts[g] for g in genre_scores}


ALGO_INFO = {
    "Most Popular": {
        "desc": "Recommends movies with the most ratings. Always picks crowd favourites.",
        "accuracy": 3, "diversity": 0, "novelty": 0, "type": "Non-personalized",
    },
    "Highest Average": {
        "desc": "Recommends highest-rated movies (min. 20 ratings). Filters out noise.",
        "accuracy": 2, "diversity": 0, "novelty": 1, "type": "Non-personalized",
    },
    "Random": {
        "desc": "Random unseen movies. Control baseline to measure if algorithms beat chance.",
        "accuracy": 0, "diversity": 3, "novelty": 3, "type": "Baseline",
    },
    "Content-Based (Genres)": {
        "desc": "Builds TF-IDF genre vectors, constructs user taste profile from centered ratings, "
                "scores by cosine similarity.",
        "accuracy": 1, "diversity": 0, "novelty": 2, "type": "Content-Based",
    },
    "Content-Based (Genres+Tags)": {
        "desc": "Same approach enriched with user-generated tags for richer item features.",
        "accuracy": 1, "diversity": 1, "novelty": 2, "type": "Content-Based",
    },
    "Item-Item CF": {
        "desc": "Adjusted cosine similarity between items. Normalised prediction from lecture formula.",
        "accuracy": 1, "diversity": 2, "novelty": 3, "type": "Collaborative",
    },
    "Matrix Factorization": {
        "desc": "Biased SGD: decomposes the rating matrix into latent factors. "
                "Discovers hidden taste dimensions.",
        "accuracy": 2, "diversity": 2, "novelty": 1, "type": "Latent Factor",
    },
}


# ─────────────────────────────────────────────────────────────────
# DATA & MODELS (cached)
# ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    ratings = load_ratings()
    items = load_items()
    tags = load_tags()
    train, test = train_test_split_ratings(ratings, test_size=0.2)
    return ratings, items, tags, train, test


@st.cache_resource
def train_models(_train, _items, _tags):
    models = {}
    for name, m in [
        ("Most Popular", MostPopularRecommender()),
        ("Highest Average", HighestAverageRatingRecommender(min_ratings=20)),
        ("Random", RandomRecommender(random_state=42)),
    ]:
        m.fit(_train)
        models[name] = m

    cb = ContentBasedRecommender(use_tags=False)
    cb.fit(_train, _items)
    models["Content-Based (Genres)"] = cb

    cb_tags = ContentBasedRecommender(use_tags=True)
    cb_tags.fit(_train, _items, tags=_tags)
    models["Content-Based (Genres+Tags)"] = cb_tags

    ii = ItemItemCollaborativeFiltering(k=30)
    ii.fit(_train)
    models["Item-Item CF"] = ii

    mf = MatrixFactorizationRecommender(
        n_factors=50, n_epochs=20, lr=0.005, reg=0.02, verbose=False
    )
    mf.fit(_train)
    models["Matrix Factorization"] = mf
    return models


with st.spinner("Loading data & training models..."):
    ratings, items, tags, train, test = load_data()
    models = train_models(train, items, tags)

all_user_ids = sorted(train[config.USER_COL].unique())
item_rating_count = train.groupby(config.ITEM_COL).size()
item_avg_rating = train.groupby(config.ITEM_COL)[config.RATING_COL].mean()


# ─────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div class="sidebar-brand">CineMatch</div>'
        '<div class="sidebar-brand-sub">Recommender Prototype</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<hr class="sec-div">', unsafe_allow_html=True)

    page = st.radio(
        "nav",
        ["Home", "Discover", "Recommendations", "User Profile",
         "Compare", "Evaluation"],
        label_visibility="collapsed",
    )

    st.markdown('<hr class="sec-div">', unsafe_allow_html=True)
    st.markdown(
        f'<div style="font-size:0.72rem; color:var(--schatten); line-height:1.7;">'
        f'<strong style="color:var(--leinwand);">Dataset</strong><br>'
        f'MovieLens Latest Small<br>'
        f'<span class="prose-num">100,836</span> ratings<br>'
        f'<span class="prose-num">9,742</span> movies<br>'
        f'<span class="prose-num">610</span> users<br><br>'
        f'<a href="https://grouplens.org/datasets/movielens/latest/" '
        f'style="color:var(--projektor);">GroupLens</a></div>',
        unsafe_allow_html=True,
    )


# ═════════════════════════════════════════════════════════════════
#  HOME
# ═════════════════════════════════════════════════════════════════
if page == "Home":
    st.markdown("# CineMatch")

    n_u = ratings[config.USER_COL].nunique()
    n_i = ratings[config.ITEM_COL].nunique()
    n_r = len(ratings)
    sparsity = 1.0 - n_r / (n_u * n_i)

    st.markdown(
        f'<p style="font-size:1.05rem; line-height:1.8; color:var(--schatten); max-width:720px;">'
        f'A recommender prototype built on the MovieLens dataset: '
        f'<span class="prose-num">{n_r:,}</span> ratings from '
        f'<span class="prose-num">{n_u:,}</span> users across '
        f'<span class="prose-num">{n_i:,}</span> movies. '
        f'The user-item matrix is <span class="prose-num">{sparsity:.1%}</span> sparse '
        f'&mdash; the core challenge every algorithm below must handle.</p>',
        unsafe_allow_html=True,
    )

    st.markdown('<hr class="sec-div">', unsafe_allow_html=True)
    st.markdown("## Algorithms")

    col1, col2 = st.columns(2, gap="medium")
    algo_list = list(ALGO_INFO.items())
    for i, (name, info) in enumerate(algo_list):
        with (col1 if i % 2 == 0 else col2):
            acc_bar = algo_prop_bar(info["accuracy"], 3, COLORS["projektor"])
            div_bar = algo_prop_bar(info["diversity"], 3, "#5b8fa8")
            nov_bar = algo_prop_bar(info["novelty"], 3, COLORS["ticket"])
            st.markdown(
                f'<div class="algo-card">'
                f'<h4>{name}</h4>'
                f'<div class="algo-desc">{info["desc"]}</div>'
                f'<div class="algo-props">'
                f'<span class="algo-prop">Accuracy {acc_bar}</span>'
                f'<span class="algo-prop">Diversity {div_bar}</span>'
                f'<span class="algo-prop">Novelty {nov_bar}</span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown('<hr class="sec-div">', unsafe_allow_html=True)
    st.markdown(
        '<div class="callout">'
        '<strong>Navigate</strong> via the sidebar to discover movies, '
        'get personalised recommendations, explore user profiles, '
        'compare algorithms side-by-side, or inspect the evaluation dashboard.'
        '</div>',
        unsafe_allow_html=True,
    )


# ═════════════════════════════════════════════════════════════════
#  DISCOVER
# ═════════════════════════════════════════════════════════════════
elif page == "Discover":
    st.markdown("## Discover Movies")

    c1, c2 = st.columns([2, 1])
    with c1:
        search = st.text_input("SEARCH", placeholder="Title, keyword...",
                                label_visibility="collapsed")
    with c2:
        all_genres = sorted(set(
            g.strip() for gs in items[config.GENRES_COL].dropna()
            for g in gs.split("|") if g.strip() and g.strip() != "(no genres listed)"
        ))
        genre_filter = st.multiselect("Genre filter", all_genres,
                                       label_visibility="collapsed",
                                       placeholder="Filter by genre...")

    sort_by = st.radio("Sort", ["Most Rated", "Highest Rated", "A \u2013 Z"],
                        horizontal=True, label_visibility="collapsed")

    filtered = items.copy()
    if search:
        filtered = filtered[filtered[config.TITLE_COL].str.contains(search, case=False, na=False)]
    if genre_filter:
        filtered = filtered[filtered[config.GENRES_COL].apply(
            lambda g: any(gf in str(g).split("|") for gf in genre_filter)
        )]

    filtered = filtered.merge(
        item_rating_count.rename("cnt"), left_on=config.ITEM_COL, right_index=True, how="left"
    ).merge(
        item_avg_rating.rename("avg"), left_on=config.ITEM_COL, right_index=True, how="left"
    )
    filtered["cnt"] = filtered["cnt"].fillna(0).astype(int)
    filtered["avg"] = filtered["avg"].fillna(0)

    if sort_by == "Most Rated":
        filtered = filtered.sort_values("cnt", ascending=False)
    elif sort_by == "Highest Rated":
        filtered = filtered[filtered["cnt"] >= 5].sort_values("avg", ascending=False)
    else:
        filtered = filtered.sort_values(config.TITLE_COL)

    st.caption(f"{len(filtered):,} movies")

    for _, row in filtered.head(25).iterrows():
        stat = (f'<em>{row["avg"]:.1f}</em> avg '
                f'\u00B7 <em>{row["cnt"]}</em> ratings')
        st.markdown(
            filmstrip_card("", row[config.TITLE_COL], row[config.GENRES_COL],
                           stat_line=stat),
            unsafe_allow_html=True,
        )

    # Similar movies
    st.markdown('<hr class="sec-div">', unsafe_allow_html=True)
    st.markdown("### Find Similar Movies")
    cb_model = models["Content-Based (Genres)"]
    movie_titles = items[config.TITLE_COL].sort_values().tolist()
    default_idx = movie_titles.index("Toy Story (1995)") if "Toy Story (1995)" in movie_titles else 0
    chosen = st.selectbox("Pick a movie", movie_titles, index=default_idx)

    if st.button("Find Similar", use_container_width=True):
        chosen_id = items.loc[items[config.TITLE_COL] == chosen, config.ITEM_COL].values
        if len(chosen_id) > 0:
            similar = cb_model.similar_items(chosen_id[0], n=6)
            if similar:
                cols = st.columns(3)
                for i, (iid, score) in enumerate(similar):
                    with cols[i % 3]:
                        t = items.loc[items[config.ITEM_COL] == iid]
                        title = t[config.TITLE_COL].values[0] if len(t) > 0 else f"Movie {iid}"
                        genre = t[config.GENRES_COL].values[0] if len(t) > 0 else ""
                        st.markdown(
                            filmstrip_card(i + 1, title, genre,
                                           stat_line=f"Cosine similarity: <em>{score:.3f}</em>"),
                            unsafe_allow_html=True,
                        )


# ═════════════════════════════════════════════════════════════════
#  RECOMMENDATIONS
# ═════════════════════════════════════════════════════════════════
elif page == "Recommendations":
    st.markdown("## Recommendations")

    c1, c2, c3 = st.columns([1, 1.5, 0.5])
    with c1:
        user_id = st.selectbox("User", all_user_ids, index=0)
    with c2:
        algo = st.selectbox("Algorithm", list(models.keys()),
                             index=list(models.keys()).index("Matrix Factorization"))
    with c3:
        n_recs = st.number_input("N", min_value=5, max_value=30, value=10)

    info = ALGO_INFO.get(algo, {})
    st.markdown(
        f'<div class="callout"><strong>{algo}</strong> '
        f'<span style="color:var(--schatten); font-size:0.78rem; '
        f'margin-left:8px; font-family:JetBrains Mono,monospace;">'
        f'{info.get("type", "")}</span><br>'
        f'{info.get("desc", "")}</div>',
        unsafe_allow_html=True,
    )

    if st.button("Generate", type="primary", use_container_width=True):
        model = models[algo]
        with st.spinner("Computing..."):
            recs = model.recommend(user_id, train, n=n_recs, exclude_seen=True)

        user_train = train[train[config.USER_COL] == user_id]
        genre_profile = get_user_genre_profile(user_train, items)
        top_genres = sorted(genre_profile.items(), key=lambda x: x[1], reverse=True)[:5]

        for rank, iid in enumerate(recs, 1):
            t_row = items.loc[items[config.ITEM_COL] == iid]
            title = t_row[config.TITLE_COL].values[0] if len(t_row) > 0 else f"Movie {iid}"
            genre = t_row[config.GENRES_COL].values[0] if len(t_row) > 0 else ""

            why = ""
            if algo == "Most Popular":
                cnt = item_rating_count.get(iid, 0)
                why = f"Rated by {cnt} users"
            elif algo == "Highest Average":
                avg = item_avg_rating.get(iid, 0)
                why = f"Average rating {avg:.2f}"
            elif algo.startswith("Content"):
                movie_genres = set(str(genre).split("|"))
                matching = [g for g, _ in top_genres if g in movie_genres]
                if matching:
                    why = f"Matches your taste: {', '.join(matching)}"
            elif algo == "Item-Item CF":
                why = "Users with similar taste also liked this"
            elif algo == "Matrix Factorization":
                if hasattr(model, "predict_score"):
                    pred = model.predict_score(user_id, iid)
                    why = f"Predicted rating: {pred:.2f}"

            st.markdown(filmstrip_card(rank, title, genre, why=why),
                        unsafe_allow_html=True)

    with st.expander("Rating History"):
        u_hist = train[train[config.USER_COL] == user_id].merge(
            items[[config.ITEM_COL, config.TITLE_COL, config.GENRES_COL]], on=config.ITEM_COL
        ).sort_values(config.RATING_COL, ascending=False)
        st.caption(f"{len(u_hist)} movies rated \u00B7 avg {u_hist[config.RATING_COL].mean():.2f}")
        for _, r in u_hist.head(12).iterrows():
            st.markdown(
                filmstrip_card(
                    "", r[config.TITLE_COL], r[config.GENRES_COL],
                    stat_line=f'<em>{r[config.RATING_COL]:.1f}</em> / 5'
                ),
                unsafe_allow_html=True,
            )


# ═════════════════════════════════════════════════════════════════
#  USER PROFILE
# ═════════════════════════════════════════════════════════════════
elif page == "User Profile":
    st.markdown("## User Profile")
    user_id = st.selectbox("User", all_user_ids, index=0)

    user_train = train[train[config.USER_COL] == user_id]
    user_items = user_train.merge(
        items[[config.ITEM_COL, config.TITLE_COL, config.GENRES_COL]], on=config.ITEM_COL
    )
    n_rated = len(user_train)
    avg_rat = user_train[config.RATING_COL].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rated", n_rated)
    c2.metric("Average", f"{avg_rat:.2f}")
    c3.metric("Min", f"{user_train[config.RATING_COL].min()}")
    c4.metric("Max", f"{user_train[config.RATING_COL].max()}")

    st.markdown('<hr class="sec-div">', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Genre Preferences")
        genre_profile = get_user_genre_profile(user_train, items)
        if genre_profile:
            gp = pd.Series(genre_profile).sort_values(ascending=True)
            fig, ax = plt.subplots(figsize=(5, max(3, len(gp) * 0.32)))
            bar_colors = [COLORS["projektor"] if v >= avg_rat else COLORS["schatten"]
                          for v in gp.values]
            gp.plot(kind="barh", ax=ax, color=bar_colors)
            ax.axvline(x=avg_rat, color=COLORS["ticket"], linestyle="--", alpha=0.6,
                       label=f"User avg ({avg_rat:.1f})")
            ax.set_xlim(0, 5.5)
            ax.set_xlabel("")
            ax.legend(fontsize=7, labelcolor=COLORS["schatten"])
            ax.tick_params(axis="y", labelsize=8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    with col2:
        st.markdown("### Rating Distribution")
        fig, ax = plt.subplots(figsize=(5, 4))
        bins = np.arange(0.25, 5.75, 0.5)
        ax.hist(user_train[config.RATING_COL], bins=bins, color=COLORS["ticket"],
                edgecolor=COLORS["vorhang"], alpha=0.85)
        ax.axvline(x=avg_rat, color=COLORS["projektor"], linestyle="--", linewidth=1.5)
        ax.set_xlabel("Rating")
        ax.set_ylabel("")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown('<hr class="sec-div">', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Favourites")
        for _, r in user_items.nlargest(6, config.RATING_COL).iterrows():
            st.markdown(
                filmstrip_card("", r[config.TITLE_COL], r[config.GENRES_COL],
                               stat_line=f'<em>{r[config.RATING_COL]:.1f}</em> / 5'),
                unsafe_allow_html=True,
            )
    with col2:
        st.markdown("### Least Liked")
        for _, r in user_items.nsmallest(6, config.RATING_COL).iterrows():
            st.markdown(
                filmstrip_card("", r[config.TITLE_COL], r[config.GENRES_COL],
                               stat_line=f'<em>{r[config.RATING_COL]:.1f}</em> / 5'),
                unsafe_allow_html=True,
            )


# ═════════════════════════════════════════════════════════════════
#  COMPARE
# ═════════════════════════════════════════════════════════════════
elif page == "Compare":
    st.markdown("## Compare Algorithms")

    user_id = st.selectbox("User", all_user_ids, index=0)
    available = list(models.keys())
    defaults = [d for d in ["Most Popular", "Content-Based (Genres)", "Matrix Factorization"]
                if d in available]
    selected = st.multiselect("Choose 2\u20134 algorithms", available,
                               default=defaults, max_selections=4)

    if len(selected) < 2:
        st.markdown(
            '<div class="callout">Select at least <strong>2</strong> algorithms to compare.</div>',
            unsafe_allow_html=True,
        )
    elif st.button("Run Comparison", type="primary", use_container_width=True):
        n = 10
        all_rec_sets = {}
        cols = st.columns(len(selected))

        for col, algo_name in zip(cols, selected):
            with col:
                st.markdown(f"### {algo_name}")
                info = ALGO_INFO.get(algo_name, {})
                st.caption(info.get("type", ""))
                with st.spinner("..."):
                    recs = models[algo_name].recommend(user_id, train, n=n, exclude_seen=True)
                all_rec_sets[algo_name] = set(recs)

                for rank, iid in enumerate(recs, 1):
                    t = items.loc[items[config.ITEM_COL] == iid]
                    title = t[config.TITLE_COL].values[0] if len(t) > 0 else f"Movie {iid}"
                    genre = t[config.GENRES_COL].values[0] if len(t) > 0 else ""
                    st.markdown(filmstrip_card(rank, title, genre), unsafe_allow_html=True)

        st.markdown('<hr class="sec-div">', unsafe_allow_html=True)
        st.markdown("### Overlap")

        algo_names = list(all_rec_sets.keys())
        n_algos = len(algo_names)
        overlap_matrix = np.zeros((n_algos, n_algos))
        for i in range(n_algos):
            for j in range(n_algos):
                overlap_matrix[i, j] = len(all_rec_sets[algo_names[i]] & all_rec_sets[algo_names[j]])

        fig, ax = plt.subplots(figsize=(max(4, n_algos * 1.5), max(3, n_algos * 1.2)))
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "cm", [COLORS["vorhang"], COLORS["projektor"]]
        )
        im = ax.imshow(overlap_matrix, cmap=cmap, vmin=0, vmax=n)
        ax.set_xticks(range(n_algos))
        ax.set_xticklabels(algo_names, rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(n_algos))
        ax.set_yticklabels(algo_names, fontsize=8)
        for i in range(n_algos):
            for j in range(n_algos):
                txt_color = COLORS["saal"] if overlap_matrix[i, j] > n * 0.5 else COLORS["leinwand"]
                ax.text(j, i, f"{int(overlap_matrix[i,j])}", ha="center", va="center",
                        fontsize=11, fontweight="bold", color=txt_color)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        total_unique = len(set.union(*all_rec_sets.values()))
        st.markdown(
            f'<div class="callout"><strong>{total_unique}</strong> unique movies '
            f'across {n * len(selected)} total slots. '
            f'{"High" if total_unique > n * len(selected) * 0.7 else "Low"} '
            f'inter-algorithm diversity.</div>',
            unsafe_allow_html=True,
        )


# ═════════════════════════════════════════════════════════════════
#  EVALUATION
# ═════════════════════════════════════════════════════════════════
elif page == "Evaluation":
    st.markdown("## Evaluation Dashboard")

    try:
        results_df = pd.read_csv(config.RESULTS_DIR / "metrics.csv", index_col=0)
    except FileNotFoundError:
        st.warning("Run `python main.py` first to generate metrics.")
        st.stop()

    st.dataframe(results_df.style.format("{:.4f}"), use_container_width=True)

    st.markdown('<hr class="sec-div">', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["Accuracy", "Beyond Accuracy", "Trade-offs", "Scalability"])

    with tab1:
        st.markdown("### Accuracy Metrics")
        acc_cols = [c for c in ["Precision@K", "Recall@K", "NDCG@K", "MRR", "Hit Rate@K"]
                    if c in results_df.columns]
        if acc_cols:
            fig, ax = plt.subplots(figsize=(12, 5))
            x = np.arange(len(results_df))
            w = 0.14
            for i, col in enumerate(acc_cols):
                ax.bar(x + i * w, results_df[col], w, label=col, color=CHART_COLORS[i % len(CHART_COLORS)])
            ax.set_xticks(x + w * (len(acc_cols) - 1) / 2)
            ax.set_xticklabels(results_df.index, rotation=25, ha="right", fontsize=8)
            ax.set_ylabel("Score")
            ax.legend(fontsize=7, labelcolor=COLORS["schatten"])
            ax.grid(axis="y")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.markdown(
            '<div class="callout">'
            '<strong>Key finding:</strong> Most Popular achieves the highest precision '
            'because it recommends crowd-favourites. '
            'Matrix Factorization is the best personalised method. '
            'CF methods struggle with 98.3% sparsity.</div>',
            unsafe_allow_html=True,
        )

    with tab2:
        st.markdown("### Beyond-Accuracy Metrics")
        beyond_cols = [c for c in ["Coverage", "Diversity", "Novelty", "Popularity Bias"]
                       if c in results_df.columns]
        if beyond_cols:
            fig, axes = plt.subplots(1, len(beyond_cols), figsize=(4 * len(beyond_cols), 4.5))
            if len(beyond_cols) == 1:
                axes = [axes]
            for ax, col in zip(axes, beyond_cols):
                vals = results_df[col]
                colors = [COLORS["projektor"] if v == vals.max() else COLORS["perf"] for v in vals]
                vals.plot(kind="barh", ax=ax, color=colors)
                ax.set_title(col, fontsize=10)
                ax.tick_params(axis="y", labelsize=7)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.markdown(
            '<div class="callout">'
            '<strong>Insight:</strong> Random achieves highest coverage (45%) and novelty. '
            'Most Popular has 100% popularity bias. '
            'Content-Based produces homogeneous lists (diversity 0.09). '
            'The ideal system would blend methods.</div>',
            unsafe_allow_html=True,
        )

    with tab3:
        st.markdown("### Trade-off Analysis")
        if all(c in results_df.columns for c in ["Precision@K", "Diversity", "Novelty"]):
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            for ax, y_col, y_label in [
                (axes[0], "Diversity", "Diversity"),
                (axes[1], "Novelty", "Novelty"),
            ]:
                for k, model_name in enumerate(results_df.index):
                    ax.scatter(
                        results_df.loc[model_name, "Precision@K"],
                        results_df.loc[model_name, y_col],
                        s=130, color=CHART_COLORS[k % len(CHART_COLORS)], zorder=5,
                    )
                    ax.annotate(
                        model_name,
                        (results_df.loc[model_name, "Precision@K"],
                         results_df.loc[model_name, y_col]),
                        textcoords="offset points", xytext=(8, 6), fontsize=7,
                        color=COLORS["schatten"],
                    )
                ax.set_xlabel("Precision@K", fontsize=9)
                ax.set_ylabel(y_label, fontsize=9)
                ax.grid(True)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.markdown(
            '<div class="callout">'
            '<strong>The core trade-off:</strong> As the lectures emphasise, '
            '"accuracy is not enough." A production system would use a hybrid approach '
            'to balance trust (accuracy) and discovery (diversity/novelty).</div>',
            unsafe_allow_html=True,
        )

    with tab4:
        st.markdown("### Scalability")
        if "Training Time (s)" in results_df.columns:
            fig, ax = plt.subplots(figsize=(10, 4))
            results_df["Training Time (s)"].sort_values().plot(
                kind="barh", ax=ax, color=COLORS["projektor"]
            )
            ax.set_xlabel("Training Time (s)", fontsize=9)
            ax.grid(axis="x")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        complexity = pd.DataFrame({
            "Algorithm": ["Random", "Most Popular", "Highest Avg",
                          "Content-Based", "Item-Item CF", "Matrix Factorization"],
            "Complexity": ["O(1)", "O(n)", "O(n)", "O(m\u00B7f)",
                           "O(u\u00B7m\u00B2)", "O(e\u00B7n\u00B7k)"],
            "Scales to": ["Any", "Millions", "Millions", "Large catalogs",
                          "Moderate", "Netflix-scale"],
        })
        st.dataframe(complexity, use_container_width=True, hide_index=True)

        st.markdown(
            '<div class="callout">'
            '<strong>Scalability:</strong> MF is expensive to train (20s) but once trained, '
            'predictions are just a dot product \u2014 O(k). '
            'CF similarity is O(n\u00B2) but cacheable. '
            'In production, MF is preferred for its offline-train / fast-predict split.</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<hr class="sec-div">', unsafe_allow_html=True)
    try:
        pred_df = pd.read_csv(config.RESULTS_DIR / "rating_prediction_metrics.csv", index_col=0)
        st.markdown("### Rating Prediction")
        st.dataframe(pred_df.style.format("{:.4f}"), use_container_width=True)
        st.caption("Evaluated on 2,000-rating test sample.")
    except FileNotFoundError:
        pass
