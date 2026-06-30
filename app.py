"""CineMatch — Movie Recommender Prototype.

Netflix-inspired UI for exploring recommendation algorithms
on the MovieLens Latest Small dataset.

Run: streamlit run app.py
"""

import re
import json
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
    train_test_split_ratings,
)
from src.baselines import (
    MostPopularRecommender, HighestAverageRatingRecommender, RandomRecommender,
)
from src.content_based import ContentBasedRecommender
from src.collaborative_filtering import ItemItemCollaborativeFiltering
from src.matrix_factorization import MatrixFactorizationRecommender
from src.poster_service import PosterService

# ─────────────────────────────────────────────────────────────────
# DESIGN TOKENS
# ─────────────────────────────────────────────────────────────────
C = {
    "bg":      "#141414",
    "surface": "#1c1c1c",
    "card":    "#232323",
    "card_h":  "#2d2d2d",
    "accent":  "#e8403e",
    "white":   "#e5e5e5",
    "grey":    "#808080",
    "dim":     "#4a4a4a",
    "overlay": "rgba(0,0,0,0.72)",
}

# Matplotlib theme
plt.rcParams.update({
    "figure.facecolor": C["bg"],
    "axes.facecolor":   C["surface"],
    "axes.edgecolor":   C["dim"],
    "axes.labelcolor":  C["white"],
    "text.color":       C["white"],
    "xtick.color":      C["grey"],
    "ytick.color":      C["grey"],
    "grid.color":       C["dim"],
    "grid.alpha":       0.25,
    "legend.facecolor": C["surface"],
    "legend.edgecolor": C["dim"],
    "font.family":      "sans-serif",
})
CHART_PALETTE = ["#e8403e", "#3b9dd9", "#5cb85c", "#f0ad4e", "#9b59b6", "#808080", "#e67e22"]

# ─────────────────────────────────────────────────────────────────
# PAGE CONFIG + GLOBAL CSS
# ─────────────────────────────────────────────────────────────────
st.set_page_config(page_title="CineMatch", page_icon="C", layout="wide",
                   initial_sidebar_state="expanded")

_CSS = """
<link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root {
    --bg: #141414;
    --surface: #1c1c1c;
    --card: #232323;
    --card-h: #2d2d2d;
    --accent: #e8403e;
    --white: #e5e5e5;
    --grey: #808080;
    --dim: #4a4a4a;
}

.stApp {
    background: var(--bg);
    color: var(--white);
    font-family: 'Inter', -apple-system, sans-serif;
}
#MainMenu, footer, header { visibility: hidden; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--surface);
    border-right: 1px solid var(--dim);
}
section[data-testid="stSidebar"] .stRadio label {
    font-family: 'Inter', sans-serif;
    font-size: 0.88rem;
    color: var(--grey);
    padding: 5px 0;
    letter-spacing: 0.2px;
}

/* Typography */
h1, h2, h3 {
    font-family: 'Bebas Neue', 'Inter', sans-serif !important;
    color: var(--white) !important;
    letter-spacing: 1.5px;
    font-weight: 400 !important;
}
h1 { font-size: 2.8rem !important; letter-spacing: 2px; }
h2 { font-size: 1.8rem !important; }
h3 { font-size: 1.3rem !important; }

/* Metrics */
[data-testid="stMetric"] {
    background: var(--surface);
    border-radius: 6px;
    padding: 10px 14px;
    border: 1px solid var(--dim);
}
[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    color: var(--white) !important;
    font-size: 1.3rem !important;
}
[data-testid="stMetricLabel"] {
    color: var(--grey) !important;
    font-size: 0.7rem !important;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* Buttons */
.stButton > button {
    background: var(--accent) !important;
    color: white !important;
    border: none !important;
    border-radius: 4px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 8px 24px !important;
    letter-spacing: 0.3px;
}
.stButton > button:hover {
    opacity: 0.85;
}

/* Tabs */
.stTabs [data-baseweb="tab"] {
    color: var(--grey);
    font-family: 'Inter', sans-serif;
    font-weight: 500;
    font-size: 0.85rem;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    color: var(--white);
}

/* Input labels */
.stSelectbox label, .stTextInput label, .stSlider label,
.stNumberInput label, .stMultiSelect label {
    color: var(--grey) !important;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}

/* Poster row (horizontal scroll) */
.poster-row {
    display: flex;
    gap: 12px;
    overflow-x: auto;
    padding: 8px 0 16px 0;
    scroll-snap-type: x mandatory;
    -webkit-overflow-scrolling: touch;
    scrollbar-width: thin;
    scrollbar-color: var(--dim) transparent;
}
.poster-row::-webkit-scrollbar { height: 6px; }
.poster-row::-webkit-scrollbar-track { background: transparent; }
.poster-row::-webkit-scrollbar-thumb { background: var(--dim); border-radius: 3px; }

/* Poster card */
.poster-card {
    flex: 0 0 160px;
    scroll-snap-align: start;
    position: relative;
    border-radius: 6px;
    overflow: hidden;
    transition: transform 0.2s ease;
    cursor: default;
}
.poster-card:hover { transform: scale(1.06); z-index: 2; }
.poster-card img {
    width: 160px; height: 240px;
    object-fit: cover; display: block; border-radius: 6px;
}
.poster-card .poster-overlay {
    position: absolute; bottom: 0; left: 0; right: 0;
    background: linear-gradient(transparent 0%, rgba(0,0,0,0.72) 40%);
    padding: 50px 10px 10px 10px;
    opacity: 0; transition: opacity 0.2s ease;
}
.poster-card:hover .poster-overlay { opacity: 1; }
.poster-card .poster-title {
    font-family: 'Inter', sans-serif; font-size: 0.78rem;
    font-weight: 600; color: white; line-height: 1.3;
}
.poster-card .poster-meta {
    font-family: 'Inter', sans-serif; font-size: 0.65rem;
    color: var(--grey); margin-top: 3px;
}
.poster-card .poster-why {
    font-size: 0.62rem; color: var(--accent);
    margin-top: 3px; font-weight: 500;
}
.poster-card .poster-rank {
    position: absolute; top: 6px; left: 8px;
    font-family: 'JetBrains Mono', monospace; font-size: 0.65rem;
    color: rgba(255,255,255,0.6); background: rgba(0,0,0,0.5);
    padding: 1px 5px; border-radius: 3px;
}

/* Poster placeholder (no image) */
.poster-placeholder {
    width: 160px; height: 240px;
    background: var(--card); border-radius: 6px;
    display: flex; align-items: center; justify-content: center; padding: 12px;
}
.poster-placeholder span {
    font-family: 'Inter', sans-serif; font-size: 0.72rem;
    color: var(--grey); text-align: center; line-height: 1.4;
}

/* Hero banner */
.hero {
    position: relative; border-radius: 8px; overflow: hidden;
    margin-bottom: 28px; min-height: 320px;
    background-size: cover; background-position: center top;
}
.hero-gradient {
    position: absolute; inset: 0;
    background: linear-gradient(to right,
        #141414 0%, rgba(20,20,20,0.85) 35%,
        rgba(20,20,20,0.4) 60%, rgba(20,20,20,0.2) 100%);
}
.hero-content {
    position: relative; padding: 48px 40px; max-width: 480px;
}
.hero-content h1 {
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 3.2rem !important; margin: 0 0 8px 0; letter-spacing: 3px;
}
.hero-content p {
    font-size: 0.92rem; color: var(--grey); line-height: 1.6; margin: 0;
}
.hero-tag {
    display: inline-block; font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem; color: var(--accent);
    text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 10px;
}

/* Row label */
.row-label {
    font-family: 'Inter', sans-serif; font-size: 1rem;
    font-weight: 600; color: var(--white); margin: 20px 0 6px 0;
}

/* Callout */
.callout {
    background: var(--surface); border-left: 3px solid var(--accent);
    padding: 12px 16px; border-radius: 0 6px 6px 0;
    margin: 12px 0; font-size: 0.82rem; color: var(--grey); line-height: 1.6;
}
.callout strong { color: var(--white); }

/* Section divider */
.sdiv { border: none; border-top: 1px solid var(--dim); margin: 24px 0; }

/* Algo list item */
.algo-item {
    background: var(--surface); border-radius: 6px;
    padding: 14px 16px; margin-bottom: 8px; border: 1px solid var(--dim);
}
.algo-item h4 {
    font-family: 'Inter', sans-serif !important; font-weight: 600;
    font-size: 0.88rem; color: var(--white) !important; margin: 0 0 4px 0;
}
.algo-item .algo-type {
    font-family: 'JetBrains Mono', monospace; font-size: 0.62rem;
    color: var(--accent); text-transform: uppercase; letter-spacing: 1px;
}
.algo-item .algo-desc {
    font-size: 0.8rem; color: var(--grey); margin-top: 6px; line-height: 1.5;
}

/* Sidebar brand */
.sidebar-brand {
    font-family: 'Bebas Neue', sans-serif; font-size: 1.8rem;
    color: var(--accent); letter-spacing: 3px;
}
.sidebar-sub {
    font-family: 'JetBrains Mono', monospace; font-size: 0.58rem;
    color: var(--grey); text-transform: uppercase;
    letter-spacing: 2px; margin-top: -4px;
}
.sidebar-stats {
    font-size: 0.72rem; color: var(--grey); line-height: 1.8;
}
</style>
"""

st.html(_CSS)


# ─────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────
def extract_year(title):
    m = re.search(r"\((\d{4})\)", str(title))
    return m.group(1) if m else ""


def clean_title(title):
    return re.sub(r"\s*\(\d{4}\)\s*$", "", str(title)).strip()


def genre_list(genres_str):
    if not genres_str or genres_str == "(no genres listed)":
        return []
    return [g.strip() for g in str(genres_str).split("|")
            if g.strip() and g.strip() != "(no genres listed)"]


def poster_card_html(movie_id, title, genres_str, poster_url,
                     rank=None, why="", show_overlay=True):
    """Render a single poster card with hover overlay."""
    year = extract_year(title)
    name = clean_title(title)
    genres = ", ".join(genre_list(genres_str)[:3])

    rank_html = f'<div class="poster-rank">{rank}</div>' if rank else ""
    why_html = f'<div class="poster-why">{why}</div>' if why else ""

    if poster_url:
        img_html = f'<img src="{poster_url}" alt="{name}" loading="lazy">'
    else:
        img_html = (f'<div class="poster-placeholder">'
                    f'<span>{name}<br>{year}</span></div>')

    overlay_html = ""
    if show_overlay:
        overlay_html = (
            f'<div class="poster-overlay">'
            f'<div class="poster-title">{name}</div>'
            f'<div class="poster-meta">{year} &middot; {genres}</div>'
            f'{why_html}</div>'
        )

    return (f'<div class="poster-card">'
            f'{rank_html}{img_html}{overlay_html}</div>')


def poster_row_html(cards_html, label=None):
    """Wrap poster cards in a horizontal scroll row with optional label."""
    label_html = f'<div class="row-label">{label}</div>' if label else ""
    return f'{label_html}<div class="poster-row">{"".join(cards_html)}</div>'


def get_user_genre_profile(user_ratings, items_df):
    merged = user_ratings.merge(items_df[[config.ITEM_COL, config.GENRES_COL]],
                                on=config.ITEM_COL)
    scores, counts = {}, {}
    for _, row in merged.iterrows():
        for g in genre_list(row[config.GENRES_COL]):
            scores[g] = scores.get(g, 0) + row[config.RATING_COL]
            counts[g] = counts.get(g, 0) + 1
    return {g: scores[g] / counts[g] for g in scores}


def get_poster(movie_id):
    """Get poster URL from cached service."""
    return posters.get_url(movie_id)


def make_rec_cards(recs, train_df, user_id, algo, model, items_df,
                   genre_profile=None, show_rank=True):
    """Build poster card HTML for a list of recommended item IDs."""
    top_genres = []
    if genre_profile:
        top_genres = sorted(genre_profile.items(), key=lambda x: x[1], reverse=True)[:5]

    cards = []
    for rank, iid in enumerate(recs, 1):
        row = items_df.loc[items_df[config.ITEM_COL] == iid]
        title = row[config.TITLE_COL].values[0] if len(row) > 0 else f"Movie {iid}"
        genre = row[config.GENRES_COL].values[0] if len(row) > 0 else ""
        url = get_poster(iid)

        why = ""
        if algo == "Most Popular":
            cnt = item_pop.get(iid, 0)
            why = f"{cnt} ratings"
        elif algo == "Highest Average":
            avg = item_avg.get(iid, 0)
            why = f"Avg {avg:.1f}"
        elif algo.startswith("Content"):
            movie_g = set(genre_list(genre))
            matching = [g for g, _ in top_genres if g in movie_g]
            if matching:
                why = ", ".join(matching[:2])
        elif algo == "Item-Item CF":
            why = "Similar taste"
        elif algo == "Matrix Factorization":
            if hasattr(model, "predict_score"):
                pred = model.predict_score(user_id, iid)
                why = f"Pred {pred:.1f}"

        cards.append(poster_card_html(
            iid, title, genre, url,
            rank=rank if show_rank else None,
            why=why,
        ))
    return cards


ALGO_INFO = {
    "Most Popular": ("Non-personalized",
        "Recommends movies with the most ratings across all users."),
    "Highest Average": ("Non-personalized",
        "Recommends highest-rated movies (minimum 20 ratings). S(u,i) = mean(r_vi)."),
    "Random": ("Baseline",
        "Recommends random unseen movies. Control baseline for evaluation."),
    "Content-Based (Genres)": ("Content-Based",
        "TF-IDF genre vectors, user profile from centered ratings, cosine similarity."),
    "Content-Based (Genres+Tags)": ("Content-Based",
        "Genre vectors enriched with user-generated tags for richer item representation."),
    "Item-Item CF": ("Collaborative Filtering",
        "Adjusted cosine item similarity. Normalised prediction: S(u,i) = r_i + ..."),
    "User-User CF": ("Collaborative Filtering",
        "Pearson correlation-based user similarity. Normalised prediction: S(u,i) = r_u + ..."),
    "Matrix Factorization": ("Latent Factor",
        "Biased SGD decomposition: r_hat = mu + b_u + b_i + p_u * q_i."),
}


# ─────────────────────────────────────────────────────────────────
# LOAD DATA & MODELS
# ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    ratings = load_ratings()
    items_df = load_items()
    tags = load_tags()
    train, test = train_test_split_ratings(ratings, test_size=0.2)
    return ratings, items_df, tags, train, test


@st.cache_resource
def train_models(_train, _items, _tags):
    models = {}
    for name, m in [
        ("Most Popular", MostPopularRecommender()),
        ("Highest Average", HighestAverageRatingRecommender(min_ratings=20)),
        ("Random", RandomRecommender(random_state=42)),
    ]:
        m.fit(_train); models[name] = m

    cb = ContentBasedRecommender(use_tags=False)
    cb.fit(_train, _items); models["Content-Based (Genres)"] = cb

    cbt = ContentBasedRecommender(use_tags=True)
    cbt.fit(_train, _items, tags=_tags); models["Content-Based (Genres+Tags)"] = cbt

    ii = ItemItemCollaborativeFiltering(k=30)
    ii.fit(_train); models["Item-Item CF"] = ii

    mf = MatrixFactorizationRecommender(n_factors=50, n_epochs=20, lr=0.005,
                                         reg=0.02, verbose=False)
    mf.fit(_train); models["Matrix Factorization"] = mf
    return models


@st.cache_resource
def load_posters():
    api_key = st.secrets.get("TMDB_API_KEY", None)
    return PosterService(api_key=api_key)


with st.spinner("Loading..."):
    ratings, items, tags, train, test = load_data()
    models = train_models(train, items, tags)
    posters = load_posters()

all_user_ids = sorted(train[config.USER_COL].unique())
item_pop = train.groupby(config.ITEM_COL).size().to_dict()
item_avg = train.groupby(config.ITEM_COL)[config.RATING_COL].mean().to_dict()


# ─────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-brand">CINEMATCH</div>'
                '<div class="sidebar-sub">Recommender Prototype</div>',
                unsafe_allow_html=True)
    st.markdown('<hr class="sdiv">', unsafe_allow_html=True)

    page = st.radio("nav",
        ["Home", "Discover", "Recommendations", "User Profile",
         "Compare", "Evaluation"],
        label_visibility="collapsed")

    st.markdown('<hr class="sdiv">', unsafe_allow_html=True)
    st.markdown(
        '<div class="sidebar-stats">'
        'MovieLens Latest Small<br>'
        '100,836 ratings<br>'
        '9,742 movies<br>'
        '610 users<br><br>'
        '<a href="https://grouplens.org/datasets/movielens/latest/" '
        'style="color:var(--accent); font-size:0.7rem;">grouplens.org</a>'
        '</div>',
        unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════
#  HOME
# ═════════════════════════════════════════════════════════════════
if page == "Home":
    # Hero: pick a well-rated popular movie with a poster
    hero_candidates = (
        train.groupby(config.ITEM_COL)
        .agg({config.RATING_COL: "mean", config.USER_COL: "count"})
        .rename(columns={config.USER_COL: "cnt", config.RATING_COL: "avg"})
        .query("cnt >= 100 and avg >= 4.0")
        .sort_values("avg", ascending=False)
    )
    hero_id = None
    hero_poster = None
    for mid in hero_candidates.index:
        url = get_poster(mid)
        if url:
            hero_id = mid
            hero_poster = url.replace("/w342/", "/w780/")
            break

    if hero_id is not None:
        hero_row = items.loc[items[config.ITEM_COL] == hero_id]
        hero_title = clean_title(hero_row[config.TITLE_COL].values[0])
        hero_year = extract_year(hero_row[config.TITLE_COL].values[0])
        hero_genres = ", ".join(genre_list(hero_row[config.GENRES_COL].values[0])[:3])
        st.markdown(
            f'<div class="hero" style="background-image:url(\'{hero_poster}\');">'
            f'<div class="hero-gradient"></div>'
            f'<div class="hero-content">'
            f'<div class="hero-tag">Featured</div>'
            f'<h1>{hero_title}</h1>'
            f'<p>{hero_year} &middot; {hero_genres}<br><br>'
            f'CineMatch explores 7 recommendation algorithms on 100,836 ratings. '
            f'Discover how different methods find different films for you.</p>'
            f'</div></div>',
            unsafe_allow_html=True)
    else:
        st.markdown("# CINEMATCH")
        st.markdown("Recommender prototype exploring 7 algorithms on MovieLens.")

    # Row: Most Popular
    pop_ids = models["Most Popular"].recommend(1, train, n=20, exclude_seen=False)
    pop_cards = []
    for iid in pop_ids:
        row = items.loc[items[config.ITEM_COL] == iid]
        t = row[config.TITLE_COL].values[0] if len(row) > 0 else ""
        g = row[config.GENRES_COL].values[0] if len(row) > 0 else ""
        pop_cards.append(poster_card_html(iid, t, g, get_poster(iid)))
    st.markdown(poster_row_html(pop_cards, label="Trending Now"), unsafe_allow_html=True)

    # Row: Highest Rated
    avg_ids = models["Highest Average"].recommend(1, train, n=20, exclude_seen=False)
    avg_cards = []
    for iid in avg_ids:
        row = items.loc[items[config.ITEM_COL] == iid]
        t = row[config.TITLE_COL].values[0] if len(row) > 0 else ""
        g = row[config.GENRES_COL].values[0] if len(row) > 0 else ""
        avg_cards.append(poster_card_html(iid, t, g, get_poster(iid)))
    st.markdown(poster_row_html(avg_cards, label="Highest Rated"), unsafe_allow_html=True)

    # Algorithms section
    st.markdown('<hr class="sdiv">', unsafe_allow_html=True)
    st.markdown("## ALGORITHMS")
    col1, col2 = st.columns(2, gap="medium")
    for i, (name, (atype, desc)) in enumerate(ALGO_INFO.items()):
        with (col1 if i % 2 == 0 else col2):
            st.markdown(
                f'<div class="algo-item">'
                f'<span class="algo-type">{atype}</span>'
                f'<h4>{name}</h4>'
                f'<div class="algo-desc">{desc}</div></div>',
                unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════
#  DISCOVER
# ═════════════════════════════════════════════════════════════════
elif page == "Discover":
    st.markdown("## DISCOVER")

    c1, c2 = st.columns([2, 1])
    with c1:
        search = st.text_input("Search", placeholder="Search by title...",
                                label_visibility="collapsed")
    with c2:
        all_genres = sorted(set(
            g.strip() for gs in items[config.GENRES_COL].dropna()
            for g in gs.split("|") if g.strip() and g.strip() != "(no genres listed)"
        ))
        gf = st.multiselect("Genre", all_genres, label_visibility="collapsed",
                             placeholder="Genre filter...")

    filtered = items.copy()
    if search:
        filtered = filtered[filtered[config.TITLE_COL].str.contains(search, case=False, na=False)]
    if gf:
        filtered = filtered[filtered[config.GENRES_COL].apply(
            lambda g: any(x in str(g).split("|") for x in gf)
        )]

    filtered = filtered.merge(
        pd.Series(item_pop, name="cnt"), left_on=config.ITEM_COL, right_index=True, how="left"
    )
    filtered["cnt"] = filtered["cnt"].fillna(0).astype(int)
    filtered = filtered.sort_values("cnt", ascending=False)

    st.caption(f"{len(filtered):,} movies")

    # Show as poster grid rows (20 per row)
    display = filtered.head(60)
    for row_start in range(0, len(display), 20):
        chunk = display.iloc[row_start:row_start + 20]
        cards = []
        for _, row in chunk.iterrows():
            cards.append(poster_card_html(
                row[config.ITEM_COL], row[config.TITLE_COL],
                row[config.GENRES_COL], get_poster(row[config.ITEM_COL])
            ))
        st.markdown(poster_row_html(cards), unsafe_allow_html=True)

    # Similar movies
    st.markdown('<hr class="sdiv">', unsafe_allow_html=True)
    st.markdown("### FIND SIMILAR")
    cb = models["Content-Based (Genres)"]
    titles = items[config.TITLE_COL].sort_values().tolist()
    default_i = titles.index("Toy Story (1995)") if "Toy Story (1995)" in titles else 0
    chosen = st.selectbox("Movie", titles, index=default_i)

    if st.button("Find Similar", use_container_width=True):
        cid = items.loc[items[config.TITLE_COL] == chosen, config.ITEM_COL].values
        if len(cid) > 0:
            similar = cb.similar_items(cid[0], n=12)
            if similar:
                sim_cards = []
                for iid, score in similar:
                    r = items.loc[items[config.ITEM_COL] == iid]
                    t = r[config.TITLE_COL].values[0] if len(r) > 0 else ""
                    g = r[config.GENRES_COL].values[0] if len(r) > 0 else ""
                    sim_cards.append(poster_card_html(
                        iid, t, g, get_poster(iid),
                        why=f"Similarity {score:.2f}"))
                st.markdown(poster_row_html(sim_cards, label=f"Similar to {clean_title(chosen)}"),
                            unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════
#  RECOMMENDATIONS
# ═════════════════════════════════════════════════════════════════
elif page == "Recommendations":
    st.markdown("## RECOMMENDATIONS")

    c1, c2, c3 = st.columns([1, 1.5, 0.5])
    with c1:
        user_id = st.selectbox("User", all_user_ids, index=0)
    with c2:
        algo = st.selectbox("Algorithm", list(models.keys()),
                             index=list(models.keys()).index("Matrix Factorization"))
    with c3:
        n_recs = st.number_input("N", min_value=5, max_value=30, value=15)

    atype, adesc = ALGO_INFO.get(algo, ("", ""))
    st.markdown(f'<div class="callout"><strong>{algo}</strong> '
                f'<span style="color:var(--accent); font-size:0.7rem; '
                f'font-family:JetBrains Mono,monospace; margin-left:6px;">'
                f'{atype}</span><br>{adesc}</div>', unsafe_allow_html=True)

    if st.button("Generate", type="primary", use_container_width=True):
        model = models[algo]
        with st.spinner("Computing..."):
            recs = model.recommend(user_id, train, n=n_recs, exclude_seen=True)

        user_train = train[train[config.USER_COL] == user_id]
        gp = get_user_genre_profile(user_train, items)
        cards = make_rec_cards(recs, train, user_id, algo, model, items, gp)
        st.markdown(poster_row_html(cards, label=f"Top {n_recs} for User {user_id}"),
                    unsafe_allow_html=True)

    with st.expander("Rating History"):
        u_hist = train[train[config.USER_COL] == user_id].merge(
            items[[config.ITEM_COL, config.TITLE_COL, config.GENRES_COL]], on=config.ITEM_COL
        ).sort_values(config.RATING_COL, ascending=False)
        st.caption(f"{len(u_hist)} movies rated")

        hist_cards = []
        for _, r in u_hist.head(20).iterrows():
            hist_cards.append(poster_card_html(
                r[config.ITEM_COL], r[config.TITLE_COL], r[config.GENRES_COL],
                get_poster(r[config.ITEM_COL]),
                why=f"{r[config.RATING_COL]:.1f} / 5"))
        st.markdown(poster_row_html(hist_cards), unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════
#  USER PROFILE
# ═════════════════════════════════════════════════════════════════
elif page == "User Profile":
    st.markdown("## USER PROFILE")
    user_id = st.selectbox("User", all_user_ids, index=0)

    ut = train[train[config.USER_COL] == user_id]
    ui = ut.merge(items[[config.ITEM_COL, config.TITLE_COL, config.GENRES_COL]],
                  on=config.ITEM_COL)
    avg_r = ut[config.RATING_COL].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rated", len(ut))
    c2.metric("Average", f"{avg_r:.2f}")
    c3.metric("Min", f"{ut[config.RATING_COL].min()}")
    c4.metric("Max", f"{ut[config.RATING_COL].max()}")

    st.markdown('<hr class="sdiv">', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### GENRE PREFERENCES")
        gp = get_user_genre_profile(ut, items)
        if gp:
            gps = pd.Series(gp).sort_values(ascending=True)
            fig, ax = plt.subplots(figsize=(5, max(3, len(gps) * 0.32)))
            bar_c = [C["accent"] if v >= avg_r else C["dim"] for v in gps.values]
            gps.plot(kind="barh", ax=ax, color=bar_c)
            ax.axvline(x=avg_r, color=C["white"], linestyle="--", alpha=0.4,
                       label=f"avg ({avg_r:.1f})")
            ax.set_xlim(0, 5.5)
            ax.set_xlabel("")
            ax.legend(fontsize=7, labelcolor=C["grey"])
            ax.tick_params(axis="y", labelsize=8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    with col2:
        st.markdown("### RATING DISTRIBUTION")
        fig, ax = plt.subplots(figsize=(5, 4))
        bins = np.arange(0.25, 5.75, 0.5)
        ax.hist(ut[config.RATING_COL], bins=bins, color=C["accent"],
                edgecolor=C["bg"], alpha=0.85)
        ax.axvline(x=avg_r, color=C["white"], linestyle="--", linewidth=1.5, alpha=0.5)
        ax.set_xlabel("Rating")
        ax.set_ylabel("")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown('<hr class="sdiv">', unsafe_allow_html=True)

    # Favourites as poster row
    top = ui.nlargest(15, config.RATING_COL)
    top_cards = []
    for _, r in top.iterrows():
        top_cards.append(poster_card_html(
            r[config.ITEM_COL], r[config.TITLE_COL], r[config.GENRES_COL],
            get_poster(r[config.ITEM_COL]),
            why=f"{r[config.RATING_COL]:.1f} / 5"))
    st.markdown(poster_row_html(top_cards, label="Favourites"), unsafe_allow_html=True)

    bot = ui.nsmallest(15, config.RATING_COL)
    bot_cards = []
    for _, r in bot.iterrows():
        bot_cards.append(poster_card_html(
            r[config.ITEM_COL], r[config.TITLE_COL], r[config.GENRES_COL],
            get_poster(r[config.ITEM_COL]),
            why=f"{r[config.RATING_COL]:.1f} / 5"))
    st.markdown(poster_row_html(bot_cards, label="Least Liked"), unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════
#  COMPARE
# ═════════════════════════════════════════════════════════════════
elif page == "Compare":
    st.markdown("## COMPARE ALGORITHMS")

    user_id = st.selectbox("User", all_user_ids, index=0)
    available = list(models.keys())
    defaults = [d for d in ["Most Popular", "Content-Based (Genres)", "Matrix Factorization"]
                if d in available]
    selected = st.multiselect("Algorithms", available, default=defaults, max_selections=4)

    if len(selected) < 2:
        st.markdown('<div class="callout">Select at least <strong>2</strong> algorithms.</div>',
                    unsafe_allow_html=True)
    elif st.button("Compare", type="primary", use_container_width=True):
        user_train = train[train[config.USER_COL] == user_id]
        gp = get_user_genre_profile(user_train, items)
        all_rec_sets = {}

        for algo_name in selected:
            model = models[algo_name]
            recs = model.recommend(user_id, train, n=10, exclude_seen=True)
            all_rec_sets[algo_name] = set(recs)
            cards = make_rec_cards(recs, train, user_id, algo_name, model, items, gp)
            st.markdown(poster_row_html(cards, label=algo_name), unsafe_allow_html=True)

        # Overlap heatmap
        st.markdown('<hr class="sdiv">', unsafe_allow_html=True)
        st.markdown("### OVERLAP")
        names = list(all_rec_sets.keys())
        n_a = len(names)
        om = np.zeros((n_a, n_a))
        for i in range(n_a):
            for j in range(n_a):
                om[i, j] = len(all_rec_sets[names[i]] & all_rec_sets[names[j]])

        fig, ax = plt.subplots(figsize=(3.5, 3))
        cmap = mcolors.LinearSegmentedColormap.from_list("c", [C["surface"], C["accent"]])
        ax.imshow(om, cmap=cmap, vmin=0, vmax=10)
        ax.set_xticks(range(n_a))
        ax.set_xticklabels(names, rotation=25, ha="right", fontsize=7)
        ax.set_yticks(range(n_a))
        ax.set_yticklabels(names, fontsize=7)
        for i in range(n_a):
            for j in range(n_a):
                tc = C["bg"] if om[i, j] > 5 else C["white"]
                ax.text(j, i, f"{int(om[i, j])}", ha="center", va="center",
                        fontsize=11, fontweight="bold", color=tc)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=False)
        plt.close()

        total_unique = len(set.union(*all_rec_sets.values()))
        st.markdown(
            f'<div class="callout"><strong>{total_unique}</strong> unique movies '
            f'across {10 * len(selected)} slots.</div>',
            unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════
#  EVALUATION
# ═════════════════════════════════════════════════════════════════
elif page == "Evaluation":
    st.markdown("## EVALUATION")

    try:
        results_df = pd.read_csv(config.RESULTS_DIR / "metrics.csv", index_col=0)
    except FileNotFoundError:
        st.warning("Run `python run_evaluation.py` first to generate metrics.")
        st.stop()

    st.dataframe(results_df.style.format("{:.4f}"), use_container_width=True)
    st.markdown('<hr class="sdiv">', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Accuracy", "Beyond Accuracy", "Trade-offs", "Fairness", "Scalability"])

    with tab1:
        st.markdown("### ACCURACY METRICS")
        acc = [c for c in ["Precision@K", "Recall@K", "NDCG@K", "MRR", "Hit Rate@K"]
               if c in results_df.columns]
        if acc:
            fig, ax = plt.subplots(figsize=(12, 5))
            x = np.arange(len(results_df))
            w = 0.14
            for i, col in enumerate(acc):
                ax.bar(x + i * w, results_df[col], w, label=col,
                       color=CHART_PALETTE[i % len(CHART_PALETTE)])
            ax.set_xticks(x + w * (len(acc) - 1) / 2)
            ax.set_xticklabels(results_df.index, rotation=25, ha="right", fontsize=8)
            ax.set_ylabel("Score")
            ax.legend(fontsize=7, labelcolor=C["grey"])
            ax.grid(axis="y")
            plt.tight_layout()
            st.pyplot(fig); plt.close()
        st.markdown(
            '<div class="callout"><strong>Key finding:</strong> Most Popular achieves highest '
            'precision (crowd favourites). Matrix Factorization is the best personalised method. '
            'CF struggles with 98.3% sparsity.</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown("### BEYOND ACCURACY")
        bc = [c for c in ["Coverage", "Diversity", "Novelty", "Popularity Bias"]
              if c in results_df.columns]
        if bc:
            fig, axes = plt.subplots(1, len(bc), figsize=(4 * len(bc), 4.5))
            if len(bc) == 1: axes = [axes]
            for ax, col in zip(axes, bc):
                vals = results_df[col]
                colors = [C["accent"] if v == vals.max() else C["dim"] for v in vals]
                vals.plot(kind="barh", ax=ax, color=colors)
                ax.set_title(col, fontsize=10)
                ax.tick_params(axis="y", labelsize=7)
            plt.tight_layout()
            st.pyplot(fig); plt.close()
        st.markdown(
            '<div class="callout"><strong>Insight:</strong> Random has highest coverage (45%) and '
            'novelty. Most Popular has 100% popularity bias. Content-Based produces homogeneous '
            'lists (diversity 0.09). The ideal system blends methods.</div>',
            unsafe_allow_html=True)

    with tab3:
        st.markdown("### TRADE-OFFS")
        if all(c in results_df.columns for c in ["Precision@K", "Diversity", "Novelty"]):
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            for ax, y_col in [(axes[0], "Diversity"), (axes[1], "Novelty")]:
                for k, mn in enumerate(results_df.index):
                    ax.scatter(results_df.loc[mn, "Precision@K"],
                               results_df.loc[mn, y_col], s=130,
                               color=CHART_PALETTE[k % len(CHART_PALETTE)], zorder=5)
                    ax.annotate(mn, (results_df.loc[mn, "Precision@K"],
                                     results_df.loc[mn, y_col]),
                                textcoords="offset points", xytext=(8, 6), fontsize=7,
                                color=C["grey"])
                ax.set_xlabel("Precision@K", fontsize=9)
                ax.set_ylabel(y_col, fontsize=9)
                ax.grid(True)
            plt.tight_layout()
            st.pyplot(fig); plt.close()
        st.markdown(
            '<div class="callout"><strong>Core trade-off:</strong> As lectures emphasise, '
            '"accuracy is not enough." A production system uses hybrid approaches to balance '
            'trust (accuracy) and discovery (diversity/novelty).</div>',
            unsafe_allow_html=True)

    with tab4:
        st.markdown("### FAIRNESS")
        fair_cols = [c for c in ["NDCG_heavy", "NDCG_light", "Fairness Gap"]
                     if c in results_df.columns]
        if fair_cols:
            fig, ax = plt.subplots(figsize=(12, 5))
            x = np.arange(len(results_df))
            w = 0.25
            if "NDCG_heavy" in results_df.columns:
                ax.bar(x - w/2, results_df["NDCG_heavy"], w, label="Heavy raters",
                       color=CHART_PALETTE[0])
            if "NDCG_light" in results_df.columns:
                ax.bar(x + w/2, results_df["NDCG_light"], w, label="Light raters",
                       color=CHART_PALETTE[1])
            ax.set_xticks(x)
            ax.set_xticklabels(results_df.index, rotation=25, ha="right", fontsize=8)
            ax.set_ylabel("NDCG@K")
            ax.legend(fontsize=8, labelcolor=C["grey"])
            ax.grid(axis="y")
            plt.tight_layout()
            st.pyplot(fig); plt.close()

            st.dataframe(results_df[fair_cols].style.format("{:.4f}"),
                         use_container_width=True)
            st.markdown(
                '<div class="callout"><strong>Fairness insight:</strong> '
                'Most Popular and Matrix Factorization show the largest gap between heavy '
                'and light raters. CB+Tags and Random are the most equitable.</div>',
                unsafe_allow_html=True)
        else:
            st.info("No fairness data. Re-run evaluation to generate.")

    with tab5:
        st.markdown("### SCALABILITY")
        if "Training Time (s)" in results_df.columns:
            fig, ax = plt.subplots(figsize=(10, 4))
            results_df["Training Time (s)"].sort_values().plot(
                kind="barh", ax=ax, color=C["accent"])
            ax.set_xlabel("Training Time (s)", fontsize=9)
            ax.grid(axis="x")
            plt.tight_layout()
            st.pyplot(fig); plt.close()
        try:
            scal_df = pd.read_csv(config.RESULTS_DIR / "scalability.csv")
            st.dataframe(scal_df, use_container_width=True, hide_index=True)
        except FileNotFoundError:
            st.dataframe(pd.DataFrame({
                "Algorithm": list(results_df.index),
                "Training Time (s)": results_df.get("Training Time (s)", pd.Series()).values,
            }), use_container_width=True, hide_index=True)

    st.markdown('<hr class="sdiv">', unsafe_allow_html=True)
    try:
        pred = pd.read_csv(config.RESULTS_DIR / "rating_prediction_metrics.csv", index_col=0)
        st.markdown("### RATING PREDICTION")
        st.dataframe(pred.style.format("{:.4f}"), use_container_width=True)
    except FileNotFoundError:
        pass
