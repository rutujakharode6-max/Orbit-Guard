import pytest
import numpy as np

from echelon_orbit.collision_prob import CollisionProbability

def test_diagonalization():
    cov = np.array([
        [0.04, 0.01],
        [0.01, 0.09]
    ])
    sx, sy, theta = CollisionProbability.diagonalize_2d_covariance(cov)
    assert sx >= sy
    assert np.isclose(sx**2 + sy**2, np.trace(cov), atol=1e-5)
    assert np.isclose((sx*sy)**2, np.linalg.det(cov), atol=1e-5)

def test_pc_foster_direct_hit():
    # Zero miss distance, small covariance -> Pc is high
    miss = np.array([0.0, 0.0])
    cov = np.array([[0.01, 0.0], [0.0, 0.01]])
    hbr = 0.005  # 5 meters
    pc = CollisionProbability.compute_pc_foster(miss, cov, hbr)
    assert pc > 0.0
    assert pc <= 1.0

def test_pc_foster_far_miss():
    # 10 km miss distance -> Pc should be practically zero
    miss = np.array([10.0, 0.0])
    cov = np.array([[0.01, 0.0], [0.0, 0.01]])
    hbr = 0.005
    pc = CollisionProbability.compute_pc_foster(miss, cov, hbr)
    assert pc < 1e-15

def test_dilution_analysis():
    # Miss of 0.2 km, very large covariance -> should detect dilution
    miss = np.array([0.2, 0.0])
    cov = np.array([[25.0, 0.0], [0.0, 25.0]])  # huge uncertainty
    hbr = 0.005
    res = CollisionProbability.analyze_dilution(miss, cov, hbr)
    assert "nominal_pc" in res
    assert "max_pc" in res
    assert res["max_pc"] >= res["nominal_pc"]
    assert res["optimal_scale"] < 1.0  # shrinking covariance increases Pc
