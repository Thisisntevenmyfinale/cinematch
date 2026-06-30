"""TMDb poster URL service with lazy loading and local JSON cache.

Fetches poster URLs from The Movie Database (TMDb) API using tmdbId
from MovieLens links.csv. Uses lazy loading: posters are fetched on
first request and cached locally in JSON for subsequent access.

Usage:
    from src.poster_service import PosterService
    posters = PosterService(api_key="your_key")
    url = posters.get_url(movie_id)  # returns URL string or None
"""

import json
import pandas as pd
import requests
from pathlib import Path
from . import config

CACHE_PATH = config.PROCESSED_DATA_DIR / "poster_urls.json"
LINKS_PATH = config.RAW_DATA_DIR / "links.csv"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w342"
TMDB_API_URL = "https://api.themoviedb.org/3/movie"


class PosterService:
    """Provides poster URLs for MovieLens movies via TMDb.

    Lazy loading: if a poster is not in the local cache and an API key
    is configured, it fetches the poster URL from TMDb on demand and
    saves it to the cache for future requests.
    """

    def __init__(self, api_key=None):
        self._cache = {}
        self._api_key = api_key
        self._links = None
        self._dirty = False
        self._load_cache()

    def _load_cache(self):
        if CACHE_PATH.exists():
            with open(CACHE_PATH, "r") as f:
                self._cache = json.load(f)

    def _save_cache(self):
        if not self._dirty:
            return
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE_PATH, "w") as f:
            json.dump(self._cache, f)
        self._dirty = False

    def _load_links(self):
        if self._links is None:
            self._links = pd.read_csv(LINKS_PATH)
        return self._links

    def _get_tmdb_id(self, movie_id):
        links = self._load_links()
        row = links.loc[links["movieId"] == movie_id, "tmdbId"]
        if len(row) > 0 and pd.notna(row.values[0]):
            return int(row.values[0])
        return None

    def _fetch_poster_url(self, tmdb_id):
        """Fetch a single poster URL from TMDb API."""
        if not self._api_key:
            return None
        try:
            resp = requests.get(
                f"{TMDB_API_URL}/{tmdb_id}",
                params={"api_key": self._api_key},
                timeout=5,
            )
            if resp.status_code == 200:
                data = resp.json()
                poster_path = data.get("poster_path")
                if poster_path:
                    return f"{TMDB_IMAGE_BASE}{poster_path}"
            return None
        except (requests.RequestException, ValueError):
            return None

    def get_url(self, movie_id):
        """Get poster URL for a MovieLens movieId.

        Returns the cached URL if available. If not cached and an API key
        is set, fetches lazily from TMDb, caches the result, and returns it.
        """
        key = str(movie_id)

        # Return from cache (hit)
        if key in self._cache:
            val = self._cache[key]
            return val if val else None

        # Lazy fetch: look up tmdbId and call API
        if self._api_key:
            tmdb_id = self._get_tmdb_id(int(movie_id))
            if tmdb_id is not None:
                url = self._fetch_poster_url(tmdb_id)
                self._cache[key] = url
                self._dirty = True
                self._save_cache()
                return url
            else:
                self._cache[key] = None
                self._dirty = True
                self._save_cache()

        return None

    @property
    def cache_size(self):
        return len(self._cache)

    @property
    def posters_found(self):
        return sum(1 for v in self._cache.values() if v)
