import pytest
from datetime import datetime, timezone
import numpy as np

from echelon_orbit.sgp4_engine import SGP4Engine, OrbitalElements

SAMPLE_LINE1 = "1 25544U 98067A   26325.26876256  .00011119  00000+0  20575-3 0  9994"
SAMPLE_LINE2 = "2 25544  51.6319 242.7442 0007380  49.5860 310.5772 15.49528864596100"

def test_sgp4_initialization():
    engine = SGP4Engine(SAMPLE_LINE1, SAMPLE_LINE2, "ISS (ZARYA)")
    assert engine.norad_id == 25544
    assert engine.name == "ISS (ZARYA)"
    assert isinstance(engine.elements, OrbitalElements)
    assert 350.0 < engine.elements.mean_altitude_km < 480.0
    assert 51.0 < engine.elements.inclination_deg < 52.0
    assert engine.elements.eccentricity < 0.01

def test_sgp4_propagation():
    engine = SGP4Engine(SAMPLE_LINE1, SAMPLE_LINE2, "ISS (ZARYA)")
    err, r, v = engine.propagate_utc(engine.epoch_utc)
    assert err == 0
    # Altitude should be ~415 km, so radius ~6790 km
    r_norm = np.linalg.norm(r)
    assert 6650.0 < r_norm < 6900.0
    # Velocity in LEO is ~7.6 km/s
    v_norm = np.linalg.norm(v)
    assert 7.2 < v_norm < 8.0

def test_trajectory_generation():
    engine = SGP4Engine(SAMPLE_LINE1, SAMPLE_LINE2, "ISS (ZARYA)")
    pos, vel, ts = engine.generate_orbit_trajectory(engine.epoch_utc, duration_minutes=30.0, step_seconds=60.0)
    assert len(pos) == 30
    assert len(vel) == 30
    assert len(ts) == 30
