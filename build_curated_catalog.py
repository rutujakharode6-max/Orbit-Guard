"""
Parser and compiler that builds data/curated_leo_catalog.json from the pre-staged TLE files.
Extracts NORAD ID, name, regime, physical size, maneuver flag, and TLE lines.
"""

import os
import json
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
TLE_DIR = os.path.join(DATA_DIR, "raw_tles")

def parse_tle_file(filepath):
    records = []
    if not os.path.exists(filepath):
        return records
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        lines = [l.rstrip() for l in f if l.strip()]
    
    i = 0
    while i < len(lines):
        # Format can be 3 lines (Name, Line 1, Line 2) or 2 lines (Line 1, Line 2)
        if lines[i].startswith("1 ") and i + 1 < len(lines) and lines[i+1].startswith("2 "):
            name = f"OBJECT-{lines[i][2:7].strip()}"
            l1 = lines[i]
            l2 = lines[i+1]
            i += 2
        elif i + 2 < len(lines) and lines[i+1].startswith("1 ") and lines[i+2].startswith("2 "):
            name = lines[i]
            l1 = lines[i+1]
            l2 = lines[i+2]
            i += 3
        else:
            i += 1
            continue
        
        norad_id = int(l1[2:7].strip())
        bstar_str = l1[53:61].strip()
        # Parse B* scientific notation e.g. "20575-3" -> 0.20575e-3
        bstar = 0.0
        try:
            if "-" in bstar_str[1:] or "+" in bstar_str[1:]:
                # e.g. 20575-3 -> 0.20575e-3
                m = re.match(r"([+-]?\d+)([+-]\d+)", bstar_str)
                if m:
                    mant = float(m.group(1)) * 1e-5
                    exp = int(m.group(2))
                    bstar = mant * (10 ** exp)
        except Exception:
            pass

        # Mean motion revs/day
        try:
            mean_motion = float(l2[52:63].strip())
            # Approximate semi-major axis in km: a = (mu / (n_rad_s)^2)^(1/3)
            # mu = 398600.4418 km^3/s^2, n_rad_s = mean_motion * 2*pi / 86400
            n_rad_s = mean_motion * 2.0 * 3.141592653589793 / 86400.0
            a = (398600.4418 / (n_rad_s ** 2)) ** (1.0 / 3.0)
            ecc = float("0." + l2[26:33].strip())
            perigee_alt = a * (1.0 - ecc) - 6378.137
            apogee_alt = a * (1.0 + ecc) - 6378.137
            mean_alt = (perigee_alt + apogee_alt) / 2.0
            inc = float(l2[8:16].strip())
        except Exception:
            mean_alt = 500.0
            perigee_alt = 500.0
            apogee_alt = 500.0
            ecc = 0.001
            inc = 51.6

        records.append({
            "name": name,
            "norad_id": norad_id,
            "line1": l1,
            "line2": l2,
            "bstar": bstar,
            "mean_alt_km": round(mean_alt, 2),
            "perigee_alt_km": round(perigee_alt, 2),
            "apogee_alt_km": round(apogee_alt, 2),
            "eccentricity": ecc,
            "inclination_deg": inc,
        })
    return records

def build_catalog():
    curated = []
    seen_ids = set()

    # 1. Stations (Regime 1)
    st = parse_tle_file(os.path.join(TLE_DIR, "stations.tle"))
    for s in st:
        if s["norad_id"] in seen_ids: continue
        seen_ids.add(s["norad_id"])
        s["regime"] = "Regime 1: VLEO / High Drag (~380-450 km)"
        s["type"] = "Space Station / Laboratory"
        s["maneuverable"] = True
        s["hard_body_radius_m"] = 54.0 if s["norad_id"] == 25544 else 20.0
        curated.append(s)

    # 2. Planet Doves & Spire Lemurs (Regime 1 CubeSats)
    doves = parse_tle_file(os.path.join(TLE_DIR, "planet.tle"))[:15]
    for d in doves:
        if d["norad_id"] in seen_ids: continue
        seen_ids.add(d["norad_id"])
        d["regime"] = "Regime 1: VLEO / High Drag (~380-450 km)"
        d["type"] = "CubeSat (Flock / PlanetScope)"
        d["maneuverable"] = False
        d["hard_body_radius_m"] = 0.5
        curated.append(d)

    lemurs = parse_tle_file(os.path.join(TLE_DIR, "spire.tle"))[:10]
    for l in lemurs:
        if l["norad_id"] in seen_ids: continue
        seen_ids.add(l["norad_id"])
        l["regime"] = "Regime 1: VLEO / High Drag (~380-450 km)"
        l["type"] = "CubeSat (Spire Lemur)"
        l["maneuverable"] = False
        l["hard_body_radius_m"] = 0.4
        curated.append(l)

    # 3. Cosmos 2251 Debris (Regime 2)
    cosmos_deb = parse_tle_file(os.path.join(TLE_DIR, "cosmos_2251_deb.tle"))[:20]
    for c in cosmos_deb:
        if c["norad_id"] in seen_ids: continue
        seen_ids.add(c["norad_id"])
        c["regime"] = "Regime 2: Constellation Shell Debris (~500-600 km)"
        c["type"] = "Collision Debris (Cosmos 2251)"
        c["maneuverable"] = False
        c["hard_body_radius_m"] = 0.3
        curated.append(c)

    # 4. Fengyun 1C Debris (Regime 2)
    fengyun_deb = parse_tle_file(os.path.join(TLE_DIR, "fengyun_1c_deb.tle"))[:20]
    for f in fengyun_deb:
        if f["norad_id"] in seen_ids: continue
        seen_ids.add(f["norad_id"])
        f["regime"] = "Regime 2: Constellation Shell Debris (~500-600 km)"
        f["type"] = "ASAT Debris (Fengyun 1C)"
        f["maneuverable"] = False
        f["hard_body_radius_m"] = 0.25
        curated.append(f)

    # 5. Iridium 33 Debris (Regime 2)
    iridium_deb = parse_tle_file(os.path.join(TLE_DIR, "iridium_33_deb.tle"))[:15]
    for i in iridium_deb:
        if i["norad_id"] in seen_ids: continue
        seen_ids.add(i["norad_id"])
        i["regime"] = "Regime 2: Constellation Shell Debris (~500-600 km)"
        i["type"] = "Collision Debris (Iridium 33)"
        i["maneuverable"] = False
        i["hard_body_radius_m"] = 0.35
        curated.append(i)

    # 6. Mid-LEO Sun-Sync & Weather (Regime 3)
    weather = parse_tle_file(os.path.join(TLE_DIR, "weather.tle"))
    # Filter for NOAA and polar weather
    for w in weather:
        if w["norad_id"] in seen_ids: continue
        if any(tag in w["name"] for tag in ["NOAA 15", "NOAA 18", "NOAA 19", "METOP", "FENGYUN", "SUOMI", "JPSS"]):
            seen_ids.add(w["norad_id"])
            w["regime"] = "Regime 3: Mid-LEO Sun-Sync & Weather (~650-850 km)"
            w["type"] = "Weather / Earth Observation"
            w["maneuverable"] = False if "NOAA" in w["name"] else True
            w["hard_body_radius_m"] = 2.5
            curated.append(w)
            if len([x for x in curated if x["regime"].startswith("Regime 3")]) >= 20:
                break

    # 7. Spent Rocket Bodies (Regime 4)
    sl16 = parse_tle_file(os.path.join(TLE_DIR, "sl16_rb.tle"))[:10]
    for s in sl16:
        if s["norad_id"] in seen_ids: continue
        seen_ids.add(s["norad_id"])
        s["regime"] = "Regime 4: Decaying & Spent Rocket Bodies"
        s["type"] = "Spent Rocket Upper Stage (SL-16 R/B)"
        s["maneuverable"] = False
        s["hard_body_radius_m"] = 5.5
        curated.append(s)

    f9 = parse_tle_file(os.path.join(TLE_DIR, "falcon9_rb.tle"))[:10]
    for f in f9:
        if f["norad_id"] in seen_ids: continue
        seen_ids.add(f["norad_id"])
        f["regime"] = "Regime 4: Decaying & Spent Rocket Bodies"
        f["type"] = "Spent Rocket Stage (Falcon 9 R/B)"
        f["maneuverable"] = False
        f["hard_body_radius_m"] = 4.2
        curated.append(f)

    print(f"Compiled curated catalog with {len(curated)} objects across 4 regimes.")
    out_file = os.path.join(DATA_DIR, "curated_leo_catalog.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(curated, f, indent=2)
    print(f"Catalog saved to {out_file}")

if __name__ == "__main__":
    build_catalog()
