"""
2D Encounter-Plane Collision Probability (Pc) & Dilution Region Analysis
Implements Foster & Alfano formulation with:
- Rotated principal axis transformation
- High-precision numerical & analytical disk integration
- Covariance dilution curve analysis (Pc vs. uncertainty scale)
- Detection of the "Dilution Region" / Dilution Paradox
"""

import math
from typing import Dict, Tuple, List
import numpy as np
from scipy import integrate

class CollisionProbability:
    """
    Computes Probability of Collision (Pc) in the 2D encounter plane
    and assesses covariance dilution effects.
    """

    @staticmethod
    def diagonalize_2d_covariance(cov_2d: np.ndarray) -> Tuple[float, float, float]:
        """
        Diagonalizes 2x2 covariance matrix into principal variances and orientation angle.
        Returns:
            sigma_x: major principal standard deviation (km)
            sigma_y: minor principal standard deviation (km)
            theta_rad: orientation angle of major axis relative to encounter frame (radians)
        """
        cxx = float(cov_2d[0, 0])
        cyy = float(cov_2d[1, 1])
        cxy = float(cov_2d[0, 1])

        # Eigenvalue calculation
        trace = cxx + cyy
        diff = cxx - cyy
        det = cxx * cyy - cxy ** 2
        term = math.sqrt(max(0.0, diff ** 2 + 4.0 * (cxy ** 2)))

        eig1 = 0.5 * (trace + term)
        eig2 = 0.5 * (trace - term)

        sigma_x = math.sqrt(max(1e-12, eig1))
        sigma_y = math.sqrt(max(1e-12, eig2))

        # Angle of major eigenvector
        if abs(cxy) > 1e-12 or abs(diff) > 1e-12:
            theta_rad = 0.5 * math.atan2(2.0 * cxy, diff)
        else:
            theta_rad = 0.0

        return sigma_x, sigma_y, theta_rad

    @classmethod
    def compute_pc_foster(
        cls,
        miss_2d: np.ndarray,
        cov_2d: np.ndarray,
        hbr_km: float,
        numerical: bool = False
    ) -> float:
        """
        Computes 2D encounter collision probability Pc.
        Uses Foster/Alfano formulation:
          - Transforms miss vector into covariance principal axes
          - Integrates bivariate normal PDF over circular disk of radius hbr_km
        """
        sigma_x, sigma_y, theta = cls.diagonalize_2d_covariance(cov_2d)

        # Rotate miss vector into principal axes
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        xm = cos_t * miss_2d[0] + sin_t * miss_2d[1]
        ym = -sin_t * miss_2d[0] + cos_t * miss_2d[1]

        # For typical LEO conjunctions, hbr_km << min(sigma_x, sigma_y)
        # Disk area approximation gives analytical closed-form:
        # Pc ~= (pi * R_c^2) / (2*pi * sigma_x * sigma_y) * exp(-0.5 * [(xm/sx)^2 + (ym/sy)^2])
        d_mahalanobis_sq = (xm / sigma_x) ** 2 + (ym / sigma_y) ** 2
        
        # Area-based analytical formula
        pc_approx = (hbr_km ** 2) / (2.0 * sigma_x * sigma_y) * math.exp(-0.5 * d_mahalanobis_sq)

        if not numerical or pc_approx < 1e-8 or pc_approx > 0.95:
            # Clamp to [0, 1]
            return float(min(1.0, max(0.0, pc_approx)))

        # High-precision numerical integration over polar disk:
        # r in [0, hbr_km], phi in [0, 2*pi]
        def integrand(phi, r):
            x = xm + r * math.cos(phi)
            y = ym + r * math.sin(phi)
            exp_term = math.exp(-0.5 * ((x / sigma_x) ** 2 + (y / sigma_y) ** 2))
            return (r / (2.0 * math.pi * sigma_x * sigma_y)) * exp_term

        try:
            val, _ = integrate.dblquad(integrand, 0.0, hbr_km, lambda r: 0.0, lambda r: 2.0 * math.pi)
            return float(min(1.0, max(0.0, val)))
        except Exception:
            return float(min(1.0, max(0.0, pc_approx)))

    @classmethod
    def analyze_dilution(
        cls,
        miss_2d: np.ndarray,
        cov_2d: np.ndarray,
        hbr_km: float,
        num_points: int = 50
    ) -> Dict:
        """
        Analyzes the Dilution Paradox:
        Sweeps scale factor lambda in [0.01, 50.0] such that Cov(lambda) = lambda^2 * Cov_2d.
        Finds:
          - nominal_pc (at lambda = 1.0)
          - max_pc (theoretical maximum collision risk if uncertainty were resolved)
          - lambda_max (scale factor where Pc peaks)
          - is_in_dilution_region: True if operational uncertainty is on the diluted tail (lambda_nom > lambda_max)
        """
        scale_factors = np.logspace(-1.5, 1.7, num_points)
        pc_curve = []

        max_pc = 0.0
        best_scale = 1.0
        nominal_pc = cls.compute_pc_foster(miss_2d, cov_2d, hbr_km)

        for s in scale_factors:
            scaled_cov = (s ** 2) * cov_2d
            p = cls.compute_pc_foster(miss_2d, scaled_cov, hbr_km)
            pc_curve.append({"scale": float(s), "pc": float(p)})
            if p > max_pc:
                max_pc = p
                best_scale = float(s)

        # In dilution region if nominal operating scale (1.0) is greater than scale of peak Pc
        # AND miss distance is small enough that high max_pc exists
        is_diluted = (best_scale < 0.85) and (max_pc > 1e-4)

        return {
            "nominal_pc": nominal_pc,
            "max_pc": max_pc,
            "optimal_scale": best_scale,
            "is_in_dilution_region": is_diluted,
            "dilution_ratio": (max_pc / max(1e-15, nominal_pc)),
            "curve_data": pc_curve
        }
