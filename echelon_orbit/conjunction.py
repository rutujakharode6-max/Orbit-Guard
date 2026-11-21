"""
Conjunction Assessment & Time of Closest Approach (TCA) Finder
Identifies close approaches between space objects using SGP4 propagation and numerical minimization.
Projects combined states and ML-calibrated uncertainties onto the 2D encounter plane.
"""

from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from typing import Tuple, Optional, Dict
import numpy as np
from scipy.optimize import minimize_scalar

from .sgp4_engine import SGP4Engine
from .frames import (
    rtn_to_teme_covariance,
    encounter_plane_projection
)
from .uncertainty_model import UncertaintyModel

@dataclass
class ConjunctionResult:
    primary_name: str
    primary_norad: int
    secondary_name: str
    secondary_norad: int
    tca_utc: datetime
    miss_distance_km: float
    relative_velocity_km_s: float
    r1_teme: np.ndarray
    v1_teme: np.ndarray
    r2_teme: np.ndarray
    v2_teme: np.ndarray
    cov1_rtn: np.ndarray
    cov2_rtn: np.ndarray
    cov1_teme: np.ndarray
    cov2_teme: np.ndarray
    miss_2d_encounter: np.ndarray
    cov_2d_encounter: np.ndarray
    hard_body_radius_m: float
    combined_hard_body_radius_km: float

class ConjunctionAssessment:
    """
    Evaluates close approach between two orbiting bodies.
    Finds exact TCA and projects combined state & uncertainty into the encounter plane.
    """
    def __init__(self, uncertainty_model: Optional[UncertaintyModel] = None):
        self.unc_model = uncertainty_model or UncertaintyModel()

    def find_tca(
        self,
        engine1: SGP4Engine,
        engine2: SGP4Engine,
        start_time: datetime,
        end_time: datetime,
        coarse_step_seconds: float = 30.0
    ) -> Tuple[datetime, float]:
        """
        Two-stage TCA finder:
        Stage 1: Coarse grid evaluation to identify candidates
        Stage 2: High-precision continuous Brent minimization
        """
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=timezone.utc)
        if end_time.tzinfo is None:
            end_time = end_time.replace(tzinfo=timezone.utc)

        duration_sec = (end_time - start_time).total_seconds()
        if duration_sec <= 0:
            raise ValueError("end_time must be after start_time")

        times = np.arange(0, duration_sec, coarse_step_seconds)
        min_dist = float("inf")
        best_t_sec = 0.0

        for t_sec in times:
            t = start_time + timedelta(seconds=float(t_sec))
            _, r1, _ = engine1.propagate_utc(t)
            _, r2, _ = engine2.propagate_utc(t)
            dist = float(np.linalg.norm(r1 - r2))
            if dist < min_dist:
                min_dist = dist
                best_t_sec = float(t_sec)

        # Refine using continuous scalar minimization within +- coarse_step_seconds
        bracket_low = max(0.0, best_t_sec - coarse_step_seconds)
        bracket_high = min(duration_sec, best_t_sec + coarse_step_seconds)

        def objective_func(t_sec):
            t = start_time + timedelta(seconds=float(t_sec))
            _, r1, _ = engine1.propagate_utc(t)
            _, r2, _ = engine2.propagate_utc(t)
            return float(np.linalg.norm(r1 - r2))

        res = minimize_scalar(
            objective_func,
            bounds=(bracket_low, bracket_high),
            method='bounded',
            options={'xatol': 1e-4}
        )

        refined_t_sec = float(res.x)
        refined_tca = start_time + timedelta(seconds=refined_t_sec)
        refined_miss = float(res.fun)

        return refined_tca, refined_miss

    def evaluate_conjunction(
        self,
        engine1: SGP4Engine,
        engine2: SGP4Engine,
        tca_utc: datetime,
        hard_body_radius1_m: float = 2.0,
        hard_body_radius2_m: float = 0.5,
        f107: float = 175.0,
        kp: float = 3.0,
        is_maneuvering1: bool = False,
        is_maneuvering2: bool = False,
        delta_t_hours: Optional[float] = None
    ) -> ConjunctionResult:
        """
        Performs full conjunction analysis at specified TCA:
        - Propagates states in TEME frame
        - Computes individual RTN and Cartesian TEME covariances using trained ML model
        - Projects combined covariance into the 2D encounter plane
        """
        _, r1, v1 = engine1.propagate_utc(tca_utc)
        _, r2, v2 = engine2.propagate_utc(tca_utc)

        miss_dist = float(np.linalg.norm(r1 - r2))
        v_rel = float(np.linalg.norm(v1 - v2))

        # Propagation time from epoch in hours
        if delta_t_hours is not None:
            dt1_hours = float(delta_t_hours)
            dt2_hours = float(delta_t_hours)
        else:
            dt1_hours = abs((tca_utc - engine1.epoch_utc).total_seconds()) / 3600.0
            dt2_hours = abs((tca_utc - engine2.epoch_utc).total_seconds()) / 3600.0

        # Generate RTN covariances from trained ML model
        cov1_rtn = self.unc_model.generate_rtn_covariance(
            delta_t_hours=dt1_hours,
            altitude_km=engine1.elements.mean_altitude_km,
            bstar=engine1.elements.bstar,
            f107=f107,
            kp=kp,
            is_maneuvering=is_maneuvering1,
            perigee_altitude_km=engine1.elements.perigee_altitude_km,
            eccentricity=engine1.elements.eccentricity,
            inclination_deg=engine1.elements.inclination_deg
        )
        cov2_rtn = self.unc_model.generate_rtn_covariance(
            delta_t_hours=dt2_hours,
            altitude_km=engine2.elements.mean_altitude_km,
            bstar=engine2.elements.bstar,
            f107=f107,
            kp=kp,
            is_maneuvering=is_maneuvering2,
            perigee_altitude_km=engine2.elements.perigee_altitude_km,
            eccentricity=engine2.elements.eccentricity,
            inclination_deg=engine2.elements.inclination_deg
        )

        # Map RTN covariances to TEME
        cov1_teme = rtn_to_teme_covariance(cov1_rtn, r1, v1)
        cov2_teme = rtn_to_teme_covariance(cov2_rtn, r2, v2)

        # Project onto 2D encounter plane
        miss_2d, cov_2d, _, _ = encounter_plane_projection(
            r1=r1, v1=v1, r2=r2, v2=v2,
            cov1_teme=cov1_teme, cov2_teme=cov2_teme
        )

        combined_hbr_m = hard_body_radius1_m + hard_body_radius2_m
        combined_hbr_km = combined_hbr_m / 1000.0

        return ConjunctionResult(
            primary_name=engine1.name,
            primary_norad=engine1.norad_id,
            secondary_name=engine2.name,
            secondary_norad=engine2.norad_id,
            tca_utc=tca_utc,
            miss_distance_km=miss_dist,
            relative_velocity_km_s=v_rel,
            r1_teme=r1,
            v1_teme=v1,
            r2_teme=r2,
            v2_teme=v2,
            cov1_rtn=cov1_rtn,
            cov2_rtn=cov2_rtn,
            cov1_teme=cov1_teme,
            cov2_teme=cov2_teme,
            miss_2d_encounter=miss_2d,
            cov_2d_encounter=cov_2d,
            hard_body_radius_m=combined_hbr_m,
            combined_hard_body_radius_km=combined_hbr_km
        )

    def screen_pairwise(
        self,
        engine1: SGP4Engine,
        engine2: SGP4Engine,
        start_time: datetime,
        end_time: datetime,
        coarse_step_seconds: float = 30.0,
        hard_body_radius1_m: float = 2.0,
        hard_body_radius2_m: float = 0.5,
        f107: float = 175.0,
        kp: float = 3.0,
        is_maneuvering1: bool = False,
        is_maneuvering2: bool = False
    ) -> ConjunctionResult:
        """
        End-to-end screener:
        1. Numerical search for true minimum miss distance timestamp (TCA).
        2. Evaluates full encounter state and ML-predicted covariance at that TCA.
        """
        tca_utc, _ = self.find_tca(
            engine1=engine1,
            engine2=engine2,
            start_time=start_time,
            end_time=end_time,
            coarse_step_seconds=coarse_step_seconds
        )
        return self.evaluate_conjunction(
            engine1=engine1,
            engine2=engine2,
            tca_utc=tca_utc,
            hard_body_radius1_m=hard_body_radius1_m,
            hard_body_radius2_m=hard_body_radius2_m,
            f107=f107,
            kp=kp,
            is_maneuvering1=is_maneuvering1,
            is_maneuvering2=is_maneuvering2
        )
