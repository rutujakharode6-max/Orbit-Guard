"""
TLE / OMM Catalog Management and Data Access
Provides access to curated satellite and debris objects with offline pre-staged fallback.
"""

import os
import json
from typing import List, Dict, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
CATALOG_PATH = os.path.join(DATA_DIR, "curated_leo_catalog.json")
BENCHMARKS_PATH = os.path.join(DATA_DIR, "conjunction_benchmarks.json")
SPACE_WEATHER_PATH = os.path.join(DATA_DIR, "space_weather.json")

class TLEFetcher:
    """Manages curated satellite catalog, space weather data, and benchmarks."""

    @classmethod
    def load_catalog(cls) -> List[Dict]:
        """Loads all curated catalog objects."""
        if not os.path.exists(CATALOG_PATH):
            return []
        with open(CATALOG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    @classmethod
    def get_by_norad_id(cls, norad_id: int) -> Optional[Dict]:
        """Finds a satellite by its NORAD ID."""
        for item in cls.load_catalog():
            if item.get("norad_id") == norad_id:
                return item
        return None

    @classmethod
    def get_by_regime(cls, regime_prefix: str) -> List[Dict]:
        """Filters satellites by orbital regime."""
        return [
            item for item in cls.load_catalog()
            if item.get("regime", "").startswith(regime_prefix)
        ]

    @classmethod
    def load_benchmarks(cls) -> List[Dict]:
        """Loads pre-configured benchmark conjunction scenarios."""
        if not os.path.exists(BENCHMARKS_PATH):
            return []
        with open(BENCHMARKS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    @classmethod
    def load_space_weather(cls) -> Dict:
        """Loads space weather indices and solar condition profiles."""
        if not os.path.exists(SPACE_WEATHER_PATH):
            return {"historical_samples": [], "regime_profiles": {}}
        with open(SPACE_WEATHER_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
