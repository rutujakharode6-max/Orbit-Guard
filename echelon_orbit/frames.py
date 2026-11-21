"""
Astrodynamic Coordinate Frame Transformations
Converts between TEME (SGP4 native), RTN/RSW (error & covariance), and Encounter Plane (2D B-plane).
"""

from typing import Tuple
import numpy as np

def teme_to_rtn_matrix(r_teme: np.ndarray, v_teme: np.ndarray) -> np.ndarray:
    """
    Constructs the 3x3 orthonormal Direction Cosine Matrix (DCM) from TEME to RTN.
    Basis vectors:
      R_hat (Radial): along position vector r / ||r||
      N_hat (Cross-track/Normal): along angular momentum (r x v) / ||r x v||
      T_hat (In-track/Along-track): completes right-handed triad (N x R)
      
    Returns DCM R_mat such that:
      v_rtn = R_mat @ v_teme
    """
    r_norm = np.linalg.norm(r_teme)
    if r_norm < 1e-6:
        raise ValueError("Position vector norm is near zero, cannot compute RTN frame.")
    r_hat = r_teme / r_norm

    h_vec = np.cross(r_teme, v_teme)
    h_norm = np.linalg.norm(h_vec)
    if h_norm < 1e-6:
        # Radial orbit or zero angular momentum fallback
        n_hat = np.array([0.0, 0.0, 1.0])
    else:
        n_hat = h_vec / h_norm

    t_hat = np.cross(n_hat, r_hat)
    t_norm = np.linalg.norm(t_hat)
    if t_norm > 1e-6:
        t_hat = t_hat / t_norm

    # Rows are the unit vectors
    dcm = np.vstack([r_hat, t_hat, n_hat])
    return dcm

def teme_to_rtn(r_teme: np.ndarray, v_teme: np.ndarray, vec_teme: np.ndarray) -> np.ndarray:
    """Transforms a vector (e.g. position residual) from TEME to RTN."""
    dcm = teme_to_rtn_matrix(r_teme, v_teme)
    return dcm @ vec_teme

def rtn_to_teme_covariance(cov_rtn: np.ndarray, r_teme: np.ndarray, v_teme: np.ndarray) -> np.ndarray:
    """
    Transforms 3x3 position covariance matrix from RTN to TEME:
      Cov_TEME = DCM.T @ Cov_RTN @ DCM
    """
    dcm = teme_to_rtn_matrix(r_teme, v_teme)
    return dcm.T @ cov_rtn @ dcm

def teme_to_rtn_covariance(cov_teme: np.ndarray, r_teme: np.ndarray, v_teme: np.ndarray) -> np.ndarray:
    """
    Transforms 3x3 position covariance matrix from TEME to RTN:
      Cov_RTN = DCM @ Cov_TEME @ DCM.T
    """
    dcm = teme_to_rtn_matrix(r_teme, v_teme)
    return dcm @ cov_teme @ dcm.T

def encounter_plane_projection(
    r1: np.ndarray,
    v1: np.ndarray,
    r2: np.ndarray,
    v2: np.ndarray,
    cov1_teme: np.ndarray,
    cov2_teme: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Projects conjunction state and combined covariance onto the 2D Encounter Plane
    (plane perpendicular to the relative velocity vector at TCA).
    
    Returns:
      miss_2d: 2D relative position vector [xi, zeta] (km)
      cov_2d: 2x2 combined covariance matrix in encounter plane (km^2)
      projection_matrix: 2x3 projection matrix mapping TEME relative vectors to encounter plane
      v_rel_mag: relative velocity magnitude (km/s)
    """
    r_rel = r1 - r2
    v_rel = v1 - v2
    v_rel_mag = float(np.linalg.norm(v_rel))

    if v_rel_mag < 1e-6:
        # Near co-moving objects (rare encounter edge case)
        eta_hat = np.array([0.0, 1.0, 0.0])
    else:
        eta_hat = v_rel / v_rel_mag

    # Build encounter plane coordinate basis (xi, zeta) perpendicular to eta_hat
    # Use primary position vector r1 to define reference direction
    cross_ref = np.cross(eta_hat, r1)
    cross_norm = np.linalg.norm(cross_ref)
    
    if cross_norm < 1e-6:
        # Fallback orthogonal reference
        arb = np.array([1.0, 0.0, 0.0]) if abs(eta_hat[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        cross_ref = np.cross(eta_hat, arb)
        cross_norm = np.linalg.norm(cross_ref)

    xi_hat = cross_ref / cross_norm
    zeta_hat = np.cross(eta_hat, xi_hat)
    zeta_hat = zeta_hat / np.linalg.norm(zeta_hat)

    # 2x3 projection matrix
    p_mat = np.vstack([xi_hat, zeta_hat])  # shape (2, 3)

    # 2D relative position in encounter plane
    miss_2d = p_mat @ r_rel

    # Combined Cartesian covariance at TCA
    cov_comb = cov1_teme + cov2_teme  # shape (3, 3)

    # 2D projected covariance
    cov_2d = p_mat @ cov_comb @ p_mat.T  # shape (2, 2)

    return miss_2d, cov_2d, p_mat, v_rel_mag
