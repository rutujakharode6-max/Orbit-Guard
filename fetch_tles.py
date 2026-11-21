"""
Data pre-staging script for ECHELON LEO Conjunction Assessment.
Downloads and caches TLE sets across 4 distinct orbital regimes:
1. High-Drag / VLEO (~380-450 km): ISS, Tiangong CSS, Planet Dove / Spire Lemur CubeSats
2. Crowded Mega-Constellation & Debris (~500-600 km): Starlink batch, Cosmos-2251 & Fengyun-1C debris
3. Mid-LEO Sun-Synchronous (~650-850 km): NOAA weather satellites, Sentinel-1A/2A, Iridium
4. Decaying & Spent Rocket Bodies: SL-16 R/B, CZ-2C R/B, Falcon-9 R/B
"""

import os
import json
import urllib.request
import ssl

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
TLE_DIR = os.path.join(DATA_DIR, "raw_tles")
os.makedirs(TLE_DIR, exist_ok=True)

GROUPS_TO_FETCH = {
    "stations": "https://celestrak.org/NORAD/elements/gp.php?GROUP=stations&FORMAT=tle",
    "weather": "https://celestrak.org/NORAD/elements/gp.php?GROUP=weather&FORMAT=tle",
    "planet": "https://celestrak.org/NORAD/elements/gp.php?GROUP=planet&FORMAT=tle",
    "spire": "https://celestrak.org/NORAD/elements/gp.php?GROUP=spire&FORMAT=tle",
}

def fetch_and_cache():
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    
    for name, url in GROUPS_TO_FETCH.items():
        out_path = os.path.join(TLE_DIR, f"{name}.tle")
        print(f"Fetching {name} from {url}...")
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
            with urllib.request.urlopen(req, context=ctx, timeout=15) as resp:
                content = resp.read().decode('utf-8')
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(content)
                lines = [l for l in content.splitlines() if l.strip()]
                print(f" Saved {name}.tle: {len(lines)//3} objects.")
        except Exception as e:
            print(f" Failed to fetch {name}: {e}")

if __name__ == "__main__":
    fetch_and_cache()
