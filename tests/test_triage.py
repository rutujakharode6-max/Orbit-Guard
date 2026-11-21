import pytest
from datetime import datetime, timezone
import numpy as np

from echelon_orbit.triage import ConjunctionTriage
from echelon_orbit.conjunction import ConjunctionResult

def test_triage_ignore():
    # 5 km miss distance -> IGNORE
    res = ConjunctionResult(
        primary_name="SAT1", primary_norad=11111,
        secondary_name="SAT2", secondary_norad=22222,
        tca_utc=datetime.now(timezone.utc),
        miss_distance_km=5.0,
        relative_velocity_km_s=12.0,
        r1_teme=np.zeros(3), v1_teme=np.zeros(3),
        r2_teme=np.zeros(3), v2_teme=np.zeros(3),
        cov1_rtn=np.eye(3), cov2_rtn=np.eye(3),
        cov1_teme=np.eye(3), cov2_teme=np.eye(3),
        miss_2d_encounter=np.array([5.0, 0.0]),
        cov_2d_encounter=np.eye(2) * 0.01,
        hard_body_radius_m=2.0,
        combined_hard_body_radius_km=0.002
    )
    decision = ConjunctionTriage.evaluate(res)
    assert decision.action == "IGNORE"
    assert decision.fuel_saved_proxy_kg > 0

def test_triage_act():
    # Direct close encounter with tight covariance -> ACT
    res = ConjunctionResult(
        primary_name="SAT1", primary_norad=11111,
        secondary_name="SAT2", secondary_norad=22222,
        tca_utc=datetime.now(timezone.utc),
        miss_distance_km=0.02, # 20 meters
        relative_velocity_km_s=14.0,
        r1_teme=np.zeros(3), v1_teme=np.zeros(3),
        r2_teme=np.zeros(3), v2_teme=np.zeros(3),
        cov1_rtn=np.eye(3)*0.0001, cov2_rtn=np.eye(3)*0.0001,
        cov1_teme=np.eye(3)*0.0001, cov2_teme=np.eye(3)*0.0001,
        miss_2d_encounter=np.array([0.02, 0.0]),
        cov_2d_encounter=np.eye(2) * 0.0001,
        hard_body_radius_m=10.0,
        combined_hard_body_radius_km=0.010
    )
    decision = ConjunctionTriage.evaluate(res)
    assert decision.action == "ACT"
