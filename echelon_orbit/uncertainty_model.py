"""
Calibrated Uncertainty & Covariance Model for SGP4 Propagation
Integrates trained Gradient Boosted Quantile Regressors (HistGBRT) with orbital physics:
- Learns empirical RTN errors from historical TLE archives & space weather
- Predicts calibrated 1-sigma, 2-sigma, and 3-sigma bounds for Radial, In-Track, and Cross-Track
- Couples In-Track and Radial errors via orbital energy cross-correlation (rho_rt)
"""

import os
import math
import joblib
from typing import Dict, Tuple, Optional, Union
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "data", "trained_quantile_models.joblib")

class UncertaintyModel:
    """
    Data-driven & physics-informed uncertainty model predicting 3x3 RTN covariance.
    Loads trained HistGradientBoostingRegressor models for empirical quantiles.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self.ml_bundle = None
        self.is_ml_loaded = False
        self._load_ml_model()

    def _load_ml_model(self):
        """Attempts to load trained ML model bundle."""
        if os.path.exists(self.model_path):
            try:
                self.ml_bundle = joblib.load(self.model_path)
                self.is_ml_loaded = True
            except Exception as e:
                print(f"[Warning] Could not load ML model from {self.model_path}: {e}")
                self.is_ml_loaded = False
        else:
            self.is_ml_loaded = False

    def predict_rtn_sigmas(
        self,
        delta_t_hours: float,
        altitude_km: float,
        bstar: float,
        f107: float = 175.0,
        kp: float = 3.0,
        is_maneuvering: bool = False,
        perigee_altitude_km: Optional[float] = None,
        eccentricity: float = 0.001,
        inclination_deg: float = 51.6
    ) -> Tuple[float, float, float]:
        """
        Computes 1-sigma standard deviations in RTN coordinates [km]:
          sigma_R: Radial standard deviation
          sigma_T: In-track (along-track) standard deviation
          sigma_N: Cross-track standard deviation
        Uses trained ML model (q=0.8413 quantile) if loaded, otherwise physics-based fallback.
        """
        dt = max(0.1, float(delta_t_hours))
        alt = max(200.0, float(altitude_km))
        hp = float(perigee_altitude_km) if perigee_altitude_km is not None else alt

        if self.is_ml_loaded and self.ml_bundle is not None:
            # Prepare feature vector
            feature_row = pd.DataFrame([{
                "delta_t_hours": dt,
                "altitude_km": alt,
                "perigee_altitude_km": hp,
                "eccentricity": float(eccentricity),
                "inclination_deg": float(inclination_deg),
                "bstar": float(bstar),
                "f107": float(f107),
                "kp": float(kp),
                "is_maneuvering": int(is_maneuvering)
            }])

            try:
                models = self.ml_bundle["models"]
                sr = float(models["radial"]["1_sigma"].predict(feature_row)[0])
                st = float(models["intrack"]["1_sigma"].predict(feature_row)[0])
                sn = float(models["crosstrack"]["1_sigma"].predict(feature_row)[0])
                return max(0.010, sr), max(0.020, st), max(0.015, sn)
            except Exception:
                pass

        # Physics-based analytical fallback (Flohrer / Oltrogge scaling)
        return self._analytical_fallback_sigmas(dt, alt, bstar, f107, kp, is_maneuvering)

    def _analytical_fallback_sigmas(
        self,
        dt: float,
        alt: float,
        bstar: float,
        f107: float,
        kp: float,
        is_maneuvering: bool
    ) -> Tuple[float, float, float]:
        """Analytical fallback based on AMOS empirical literature."""
        eff_bstar = max(1e-6, abs(bstar))
        sw_factor = ((max(65.0, min(300.0, f107)) / 150.0) ** 1.35) * (1.0 + 0.08 * max(0.0, min(9.0, kp)))
        h_diff = (alt - 400.0) / 65.0
        drag_density = math.exp(-max(-2.0, min(8.0, h_diff)))

        # 1. Radial
        sigma_r = 0.025 + 0.0035 * (dt ** 0.55)
        # 2. Cross-Track
        sigma_n = 0.040 + 0.0060 * (dt ** 0.70)
        # 3. Along-Track
        drag_coeff = 260.0 * eff_bstar * drag_density * sw_factor
        sigma_t = 0.080 + 0.025 * dt + drag_coeff * ((dt / 24.0) ** 2.0)

        if is_maneuvering:
            sigma_t *= 1.35
            sigma_r *= 1.15
            sigma_n *= 1.15

        return float(sigma_r), float(sigma_t), float(sigma_n)

    def predict_quantiles(
        self,
        delta_t_hours: float,
        altitude_km: float,
        bstar: float,
        f107: float = 175.0,
        kp: float = 3.0,
        is_maneuvering: bool = False,
        perigee_altitude_km: Optional[float] = None,
        eccentricity: float = 0.001,
        inclination_deg: float = 51.6
    ) -> Dict[str, Dict[str, float]]:
        """
        Returns full quantile intervals (median, 1-sigma, 2-sigma, 3-sigma) for each axis.
        """
        dt = max(0.1, float(delta_t_hours))
        alt = max(200.0, float(altitude_km))
        hp = float(perigee_altitude_km) if perigee_altitude_km is not None else alt

        res = {"radial": {}, "intrack": {}, "crosstrack": {}}

        if self.is_ml_loaded and self.ml_bundle is not None:
            feature_row = pd.DataFrame([{
                "delta_t_hours": dt,
                "altitude_km": alt,
                "perigee_altitude_km": hp,
                "eccentricity": float(eccentricity),
                "inclination_deg": float(inclination_deg),
                "bstar": float(bstar),
                "f107": float(f107),
                "kp": float(kp),
                "is_maneuvering": int(is_maneuvering)
            }])
            models = self.ml_bundle["models"]
            for axis in ["radial", "intrack", "crosstrack"]:
                for q_lbl in ["median", "1_sigma", "2_sigma", "3_sigma"]:
                    val = float(models[axis][q_lbl].predict(feature_row)[0])
                    res[axis][q_lbl] = round(val, 4)
            return res

        # Fallback extrapolation
        sr, st, sn = self._analytical_fallback_sigmas(dt, alt, bstar, f107, kp, is_maneuvering)
        for axis, sig in [("radial", sr), ("intrack", st), ("crosstrack", sn)]:
            res[axis]["median"] = round(0.674 * sig, 4)
            res[axis]["1_sigma"] = round(sig, 4)
            res[axis]["2_sigma"] = round(2.0 * sig, 4)
            res[axis]["3_sigma"] = round(3.0 * sig, 4)
        return res

    def generate_rtn_covariance(
        self,
        delta_t_hours: float,
        altitude_km: float,
        bstar: float,
        f107: float = 175.0,
        kp: float = 3.0,
        is_maneuvering: bool = False,
        rho_rt: float = 0.30,
        perigee_altitude_km: Optional[float] = None,
        eccentricity: float = 0.001,
        inclination_deg: float = 51.6
    ) -> np.ndarray:
        """
        Constructs the full 3x3 position covariance matrix in RTN frame (units: km^2).
        Includes the physical cross-correlation between Radial and In-Track errors (rho_rt).
        """
        sr, st, sn = self.predict_rtn_sigmas(
            delta_t_hours=delta_t_hours,
            altitude_km=altitude_km,
            bstar=bstar,
            f107=f107,
            kp=kp,
            is_maneuvering=is_maneuvering,
            perigee_altitude_km=perigee_altitude_km,
            eccentricity=eccentricity,
            inclination_deg=inclination_deg
        )

        cov_rt = rho_rt * sr * st
        cov_rtn = np.array([
            [sr ** 2,    cov_rt,    0.0     ],
            [cov_rt,     st ** 2,   0.0     ],
            [0.0,        0.0,       sn ** 2 ]
        ], dtype=np.float64)

        return cov_rtn
