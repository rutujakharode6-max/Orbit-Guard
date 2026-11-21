"""
High-Precision SGP4 Analytical Orbit Propagator & State Extraction
Provides TEME Cartesian state vectors and derived orbital elements.
"""

import math
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Tuple, Optional, List, Union
import numpy as np
from sgp4.api import Satrec, jday

# Earth Physical Constants (WGS-84 / WGS-72 compatibility)
EARTH_RADIUS_KM = 6378.137  # Earth equatorial radius in km
MU_EARTH = 398600.4418      # Earth gravitational parameter km^3 / s^2

@dataclass
class OrbitalElements:
    norad_id: int
    name: str
    epoch_utc: datetime
    semi_major_axis_km: float
    perigee_altitude_km: float
    apogee_altitude_km: float
    mean_altitude_km: float
    eccentricity: float
    inclination_deg: float
    raan_deg: float
    arg_perigee_deg: float
    mean_anomaly_deg: float
    mean_motion_rev_day: float
    period_minutes: float
    bstar: float

class SGP4Engine:
    """
    Wrapper around official SGP4 propagator providing:
    - TEME state propagation (r, v)
    - Keplerian and physical element calculations
    - Orbit decay and condition verification
    """
    def __init__(self, line1: str, line2: str, name: str = "UNKNOWN"):
        self.name = name.strip()
        self.line1 = line1.strip()
        self.line2 = line2.strip()
        self.satrec = Satrec.twoline2rv(self.line1, self.line2)
        self.norad_id = int(self.satrec.satnum)
        self.epoch_utc = self._get_epoch_datetime()
        self.elements = self._extract_orbital_elements()

    def _get_epoch_datetime(self) -> datetime:
        """Extract exact epoch datetime from Satrec object in UTC."""
        epoch_jd = self.satrec.jdsatepoch + self.satrec.jdsatepochF
        # Convert Julian date to calendar date
        # Algorithm from Meeus Astronomical Algorithms
        z = int(epoch_jd + 0.5)
        f = (epoch_jd + 0.5) - z
        if z < 2299161:
            a = z
        else:
            alpha = int((z - 1867216.25) / 36524.25)
            a = z + 1 + alpha - int(alpha / 4)
        b = a + 1524
        c = int((b - 122.1) / 365.25)
        d = int(365.25 * c)
        e = int((b - d) / 30.6001)
        day = b - d - int(30.6001 * e) + f
        if e < 14:
            month = e - 1
        else:
            month = e - 13
        if month > 2:
            year = c - 4716
        else:
            year = c - 4715
        
        day_int = int(day)
        day_frac = day - day_int
        total_seconds = day_frac * 86400.0
        hour = int(total_seconds // 3600)
        minute = int((total_seconds % 3600) // 60)
        second = total_seconds % 60
        microsecond = int((second - int(second)) * 1e6)
        
        return datetime(year, month, day_int, hour, minute, int(second), microsecond, tzinfo=timezone.utc)

    def _extract_orbital_elements(self) -> OrbitalElements:
        """Compute Keplerian elements from SGP4 mean elements at epoch."""
        # Mean motion in radians per minute to revs per day
        n_rad_min = self.satrec.no_kozai
        n_rev_day = n_rad_min * (1440.0 / (2.0 * math.pi))
        n_rad_sec = n_rad_min / 60.0

        # Semi-major axis from mean motion: a = (mu / n^2)^(1/3)
        if n_rad_sec > 0:
            a = (MU_EARTH / (n_rad_sec ** 2)) ** (1.0 / 3.0)
        else:
            a = EARTH_RADIUS_KM + 500.0

        ecc = float(self.satrec.ecco)
        hp = a * (1.0 - ecc) - EARTH_RADIUS_KM
        ha = a * (1.0 + ecc) - EARTH_RADIUS_KM
        h_mean = (hp + ha) / 2.0
        period_min = 1440.0 / n_rev_day if n_rev_day > 0 else 90.0

        return OrbitalElements(
            norad_id=self.norad_id,
            name=self.name,
            epoch_utc=self.epoch_utc,
            semi_major_axis_km=round(a, 3),
            perigee_altitude_km=round(hp, 3),
            apogee_altitude_km=round(ha, 3),
            mean_altitude_km=round(h_mean, 3),
            eccentricity=round(ecc, 7),
            inclination_deg=round(math.degrees(self.satrec.inclo), 4),
            raan_deg=round(math.degrees(self.satrec.nodeo), 4),
            arg_perigee_deg=round(math.degrees(self.satrec.argpo), 4),
            mean_anomaly_deg=round(math.degrees(self.satrec.mo), 4),
            mean_motion_rev_day=round(n_rev_day, 8),
            period_minutes=round(period_min, 3),
            bstar=float(self.satrec.bstar)
        )

    def propagate_utc(self, dt: datetime) -> Tuple[int, np.ndarray, np.ndarray]:
        """
        Propagate to a target UTC datetime.
        Returns:
            error_code: 0 on success, non-zero if SGP4 warning/decay
            r_teme: 3D position vector in TEME frame (km) [x, y, z]
            v_teme: 3D velocity vector in TEME frame (km/s) [vx, vy, vz]
        """
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        
        jd, fr = jday(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second + dt.microsecond * 1e-6)
        error, r, v = self.satrec.sgp4(jd, fr)
        
        if error != 0:
            # Fallback zero vector if decayed
            return error, np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64)
            
        return error, np.array(r, dtype=np.float64), np.array(v, dtype=np.float64)

    def propagate_delta_t_hours(self, delta_hours: float) -> Tuple[int, np.ndarray, np.ndarray, datetime]:
        """Propagate state relative to epoch by delta_hours."""
        from datetime import timedelta
        target_time = self.epoch_utc + timedelta(hours=delta_hours)
        error, r, v = self.propagate_utc(target_time)
        return error, r, v, target_time

    def generate_orbit_trajectory(self, start_dt: datetime, duration_minutes: float, step_seconds: float = 60.0) -> Tuple[np.ndarray, np.ndarray, List[datetime]]:
        """
        Generate time-series trajectory vectors for 3D visualization.
        Returns:
            positions: shape (N, 3) in km
            velocities: shape (N, 3) in km/s
            timestamps: list of UTC datetimes
        """
        from datetime import timedelta
        num_steps = max(2, int((duration_minutes * 60.0) / step_seconds))
        positions = []
        velocities = []
        timestamps = []

        for step in range(num_steps):
            t = start_dt + timedelta(seconds=step * step_seconds)
            err, r, v = self.propagate_utc(t)
            if err == 0:
                positions.append(r)
                velocities.append(v)
                timestamps.append(t)

        return np.array(positions), np.array(velocities), timestamps
