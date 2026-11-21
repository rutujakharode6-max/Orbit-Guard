"""
Curates a high-quality dataset of ~80-120 representative LEO objects across 4 regimes:
1. High-Drag / VLEO (~380-450 km): ISS, Tiangong CSS, Planet Doves, Lemurs
2. Crowded Mega-Constellation Shells & Inactive Debris (~500-600 km): Starlinks, Cosmos 2251 Debris, Fengyun 1C Debris, Iridium 33 Debris
3. Mid-LEO Sun-Synchronous & Earth Obs (~650-850 km): NOAA-15, NOAA-18, NOAA-19, Sentinel-1A/2A, Iridium
4. Decaying & Spent Rocket Bodies: SL-16 R/B, Falcon 9 R/B, CZ-2C R/B
"""

import os
import urllib.request
import urllib.parse
import ssl
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
TLE_DIR = os.path.join(DATA_DIR, "raw_tles")
os.makedirs(TLE_DIR, exist_ok=True)

def fetch_named_objects(name, limit=25):
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    enc = urllib.parse.quote(name)
    url = f"https://celestrak.org/NORAD/elements/gp.php?NAME={enc}&FORMAT=tle"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=12) as resp:
            text = resp.read().decode('utf-8')
            lines = [l.strip() for l in text.splitlines() if l.strip()]
            records = []
            for i in range(0, len(lines) - 2, 3):
                records.append({
                    "name": lines[i],
                    "line1": lines[i+1],
                    "line2": lines[i+2]
                })
                if len(records) >= limit:
                    break
            return records
    except Exception as e:
        print(f"Error fetching {name}: {e}")
        return []

def main():
    print("Fetching curated objects across regimes...")
    curated = []
    
    # Regime 1: VLEO / High Drag
    print("1. Fetching Regime 1: VLEO / High-Drag...")
    stations = fetch_named_objects("ZARYA", limit=1) # ISS
    stations += fetch_named_objects("TIANGONG", limit=1) # Tiangong CSS
    planet_doves = fetch_named_objects("FLOCK", limit=10) # Planet Dove CubeSats
    lemurs = fetch_named_objects("LEMUR", limit=10) # Spire Lemur
    
    for obj in stations:
        obj["regime"] = "Regime 1: VLEO / High Drag (~380-450 km)"
        obj["type"] = "Manned Station"
        obj["maneuverable"] = True
        obj["hard_body_radius_m"] = 54.0 if "ZARYA" in obj["name"] else 20.0
        curated.append(obj)
        
    for obj in planet_doves + lemurs:
        obj["regime"] = "Regime 1: VLEO / High Drag (~380-450 km)"
        obj["type"] = "CubeSat / SmallSat"
        obj["maneuverable"] = False
        obj["hard_body_radius_m"] = 0.4
        curated.append(obj)
        
    # Regime 2: Crowded Shell & Clean Debris (500-600 km)
    print("2. Fetching Regime 2: Starlinks and Pure Debris...")
    cosmos_deb = fetch_named_objects("COSMOS 2251 DEB", limit=15)
    fengyun_deb = fetch_named_objects("FENGYUN 1C DEB", limit=15)
    iridium_deb = fetch_named_objects("IRIDIUM 33 DEB", limit=10)
    starlinks = fetch_named_objects("STARLINK", limit=15)
    
    for obj in cosmos_deb + fengyun_deb + iridium_deb:
        obj["regime"] = "Regime 2: Constellation Shell Debris (~500-600 km)"
        obj["type"] = "Debris (Non-maneuvering Ground Truth)"
        obj["maneuverable"] = False
        obj["hard_body_radius_m"] = 0.25
        curated.append(obj)
        
    for obj in starlinks:
        obj["regime"] = "Regime 2: Mega-Constellation Shell (~500-560 km)"
        obj["type"] = "Active Constellation"
        obj["maneuverable"] = True
        obj["hard_body_radius_m"] = 4.0
        curated.append(obj)
        
    # Regime 3: Mid-LEO Sun-Sync & Weather (~650-850 km)
    print("3. Fetching Regime 3: Mid-LEO Weather & Sun-Sync...")
    noaa15 = fetch_named_objects("NOAA 15", limit=1)
    noaa18 = fetch_named_objects("NOAA 18", limit=1)
    noaa19 = fetch_named_objects("NOAA 19", limit=1)
    sentinels = fetch_named_objects("SENTINEL", limit=4)
    iridium_active = fetch_named_objects("IRIDIUM", limit=10)
    
    for obj in noaa15 + noaa18 + noaa19 + sentinels:
        obj["regime"] = "Regime 3: Mid-LEO Sun-Sync & Weather (~650-850 km)"
        obj["type"] = "Earth Observation / Weather"
        obj["maneuverable"] = False if "NOAA" in obj["name"] else True
        obj["hard_body_radius_m"] = 2.5
        curated.append(obj)
        
    for obj in iridium_active:
        obj["regime"] = "Regime 3: Mid-LEO Sun-Sync & Weather (~650-850 km)"
        obj["type"] = "Communications"
        obj["maneuverable"] = True
        obj["hard_body_radius_m"] = 3.0
        curated.append(obj)
        
    # Regime 4: High-Eccentricity / Spent Rocket Bodies
    print("4. Fetching Regime 4: Spent Rocket Bodies...")
    sl16 = fetch_named_objects("SL-16 R/B", limit=8)
    falcon9 = fetch_named_objects("FALCON 9 R/B", limit=8)
    
    for obj in sl16 + falcon9:
        obj["regime"] = "Regime 4: Decaying & Spent Rocket Bodies"
        obj["type"] = "Rocket Body (Large RCS / High Drag)"
        obj["maneuverable"] = False
        obj["hard_body_radius_m"] = 5.5
        curated.append(obj)
        
    print(f"Total curated objects collected: {len(curated)}")
    catalog_path = os.path.join(DATA_DIR, "curated_leo_catalog.json")
    with open(catalog_path, "w", encoding="utf-8") as f:
        json.dump(curated, f, indent=2)
    print(f"Saved catalog to {catalog_path}")

if __name__ == "__main__":
    main()
