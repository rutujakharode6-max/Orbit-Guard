import pytest
import os
import numpy as np

# Suppress Windows joblib warning
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

from echelon_orbit.uncertainty_model import UncertaintyModel
from echelon_orbit.conjunction import ConjunctionAssessment
from echelon_orbit.sgp4_engine import SGP4Engine

def test_ml_uncertainty_loading():
    model = UncertaintyModel()
    # Check that model loads ML bundle if present or falls back cleanly
    assert hasattr(model, "is_ml_loaded")
    sig_r, sig_t, sig_n = model.predict_rtn_sigmas(
        delta_t_hours=24.0,
        altitude_km=420.0,
        bstar=0.00015,
        f107=190.0,
        kp=4.0
    )
    # Physical sanity checks
    assert 0.010 <= sig_r <= 0.300  # Radial error 10m - 300m
    assert 0.100 <= sig_t <= 5.000  # In-track drag error 100m - 5km
    assert 0.020 <= sig_n <= 0.500  # Cross-track error 20m - 500m
    # Along-track must dominate radial and cross-track
    assert sig_t > sig_r
    assert sig_t > sig_n

def test_coupled_rtn_covariance():
    model = UncertaintyModel()
    cov = model.generate_rtn_covariance(
        delta_t_hours=24.0,
        altitude_km=500.0,
        bstar=0.0001,
        rho_rt=0.30
    )
    assert cov.shape == (3, 3)
    # Symmetric
    np.testing.assert_allclose(cov, cov.T, atol=1e-10)
    # Positive definite (all eigenvalues > 0)
    eigenvalues = np.linalg.eigvals(cov)
    assert all(ev > 0 for ev in eigenvalues)
    # Check non-zero cross-correlation
    assert cov[0, 1] > 0.0
    assert cov[1, 0] == cov[0, 1]

def test_numerical_tca_optimization():
    l1_p = "1 25544U 98067A   26325.26876256  .00011119  00000+0  20575-3 0  9994"
    l2_p = "2 25544  51.6319 242.7442 0007380  49.5860 310.5772 15.49528864596100"
    eng1 = SGP4Engine(l1_p, l2_p, "ISS")
    
    # Secondary passing close
    l1_s = "1 34085U 93036ARF 26325.26876256  .00008542  00000+0  16234-3 0  9991"
    l2_s = "2 34085  51.6369 242.7442 0007380  49.5860 310.5831 15.49528864000018"
    eng2 = SGP4Engine(l1_s, l2_s, "DEB")

    from datetime import timedelta
    screener = ConjunctionAssessment()
    tca, min_d = screener.find_tca(
        engine1=eng1,
        engine2=eng2,
        start_time=eng1.epoch_utc,
        end_time=eng1.epoch_utc + timedelta(hours=2.0),
        coarse_step_seconds=15.0
    )
    assert tca >= eng1.epoch_utc
    # Physical miss distance should be sub-kilometer (~685 m)
    assert 0.500 <= min_d <= 0.850
