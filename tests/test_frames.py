import pytest
import numpy as np

from echelon_orbit.frames import (
    teme_to_rtn_matrix,
    teme_to_rtn,
    rtn_to_teme_covariance,
    teme_to_rtn_covariance,
    encounter_plane_projection
)

def test_rtn_matrix_orthonormality():
    r = np.array([5000.0, 3000.0, 2500.0])
    v = np.array([-2.0, 6.0, 3.5])
    dcm = teme_to_rtn_matrix(r, v)
    
    # Check that DCM @ DCM.T = Identity
    identity_approx = dcm @ dcm.T
    np.testing.assert_allclose(identity_approx, np.eye(3), atol=1e-10)
    
    # Determinant of right-handed rotation matrix is +1
    assert np.isclose(np.linalg.det(dcm), 1.0, atol=1e-10)

def test_covariance_roundtrip():
    r = np.array([4000.0, 4000.0, 2000.0])
    v = np.array([-4.0, 3.0, 5.0])
    
    cov_rtn = np.array([
        [0.01, 0.005, 0.0],
        [0.005, 0.25, 0.0],
        [0.0, 0.0, 0.04]
    ])
    
    cov_teme = rtn_to_teme_covariance(cov_rtn, r, v)
    cov_rtn_recovered = teme_to_rtn_covariance(cov_teme, r, v)
    
    np.testing.assert_allclose(cov_rtn, cov_rtn_recovered, atol=1e-10)

def test_encounter_plane_projection():
    r1 = np.array([6800.0, 0.0, 0.0])
    v1 = np.array([0.0, 7.6, 0.0])
    
    # Head-on relative pass with 0.5 km miss along z
    r2 = np.array([6800.0, 0.0, 0.5])
    v2 = np.array([0.0, -7.6, 0.0])
    
    cov1 = np.eye(3) * 0.05
    cov2 = np.eye(3) * 0.05
    
    miss_2d, cov_2d, p_mat, v_rel_mag = encounter_plane_projection(r1, v1, r2, v2, cov1, cov2)
    
    assert np.isclose(v_rel_mag, 15.2, atol=1e-3)
    assert np.isclose(np.linalg.norm(miss_2d), 0.5, atol=1e-3)
    assert cov_2d.shape == (2, 2)
