"""
Actionable Decision Triage & False Alarm Reduction Engine
Maps calibrated collision probability (Pc) and dilution metrics to:
- IGNORE: Low risk (< 1e-6) - dismiss benign false alarm
- MONITOR: Intermediate risk (1e-6 to 1e-4) or Dilution Warning - queue updates
- ACT: High risk (>= 1e-4) - execute collision avoidance maneuver (CAM)
"""

from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np

from .collision_prob import CollisionProbability
from .conjunction import ConjunctionResult

@dataclass
class TriageDecision:
    action: str  # "IGNORE", "MONITOR", "ACT"
    nominal_pc: float
    max_pc: float
    is_diluted: bool
    miss_distance_km: float
    hbr_m: float
    status_color: str
    rationale: str
    action_protocol: str
    fuel_saved_proxy_kg: float

class ConjunctionTriage:
    """
    Evaluates conjunction results against standard operational safety thresholds:
      - Threshold IGNORE: Pc < 1e-6
      - Threshold MONITOR: 1e-6 <= Pc < 1e-4 (or dilution flag)
      - Threshold ACT: Pc >= 1e-4
    """
    PC_IGNORE_THRESHOLD = 1e-6
    PC_ACT_THRESHOLD = 1e-4

    @classmethod
    def evaluate(
        cls,
        conjunction_res: ConjunctionResult,
        numerical_pc: bool = False
    ) -> TriageDecision:
        """Determines the operational triage decision for a conjunction event."""
        miss_2d = conjunction_res.miss_2d_encounter
        cov_2d = conjunction_res.cov_2d_encounter
        hbr_km = conjunction_res.combined_hard_body_radius_km
        miss_km = conjunction_res.miss_distance_km

        # 1. Compute nominal Pc
        nom_pc = CollisionProbability.compute_pc_foster(
            miss_2d=miss_2d,
            cov_2d=cov_2d,
            hbr_km=hbr_km,
            numerical=numerical_pc
        )

        # 2. Analyze Dilution
        dil_info = CollisionProbability.analyze_dilution(
            miss_2d=miss_2d,
            cov_2d=cov_2d,
            hbr_km=hbr_km
        )
        max_pc = dil_info["max_pc"]
        is_diluted = dil_info["is_in_dilution_region"]

        # 3. Decision Logic
        if nom_pc >= cls.PC_ACT_THRESHOLD:
            action = "ACT"
            status_color = "#E63946"  # Red
            rationale = (
                f"Calibrated Pc ({nom_pc:.2e}) exceeds the 1e-4 operational action threshold. "
                f"Close approach distance is {miss_km * 1000.0:.1f} m with high-confidence encounter geometry. "
                f"Immediate collision avoidance maneuver recommended."
            )
            action_protocol = "Queue propulsion system for collision-avoidance delta-V burn. Verify post-burn clearance."
            fuel_saved = 0.0

        elif is_diluted and (max_pc >= cls.PC_ACT_THRESHOLD):
            # Dilution override!
            action = "MONITOR"
            status_color = "#F4A261"  # Orange/Amber
            rationale = (
                f"DILUTION WARNING: Nominal Pc is {nom_pc:.2e}, but severe state uncertainty suppresses "
                f"probability density. Peak potential Pc is {max_pc:.2e}. Conjunction cannot be safely ignored."
            )
            action_protocol = "Do not maneuver yet. Queue high-priority radar/laser tracking pass to shrink covariance."
            fuel_saved = 0.0

        elif nom_pc >= cls.PC_IGNORE_THRESHOLD:
            action = "MONITOR"
            status_color = "#E9C46A"  # Amber
            rationale = (
                f"Calibrated Pc ({nom_pc:.2e}) sits in the monitoring window [1e-6, 1e-4]. "
                f"Encounter is elevated but below mandatory burn criteria."
            )
            action_protocol = "Track upcoming TLE/OMM updates at next catalog release. Prepare contingency burn plan."
            fuel_saved = 0.0

        else:
            action = "IGNORE"
            status_color = "#2A9D8F"  # Teal/Green
            rationale = (
                f"Calibrated Pc ({nom_pc:.2e}) is well below the 1e-6 threshold. "
                f"Miss distance ({miss_km * 1000.0:.1f} m) represents a safe pass under calibrated uncertainty. "
                f"Baseline screening alert safely downgraded to IGNORE."
            )
            action_protocol = "Archive event log. No operator intervention or propellant expenditure required."
            # Nominal proxy of fuel saved by avoiding an unnecessary 1 m/s delta-v maneuver on a 500kg spacecraft
            fuel_saved = 0.75  # kg hydrazine proxy

        return TriageDecision(
            action=action,
            nominal_pc=nom_pc,
            max_pc=max_pc,
            is_diluted=is_diluted,
            miss_distance_km=miss_km,
            hbr_m=conjunction_res.hard_body_radius_m,
            status_color=status_color,
            rationale=rationale,
            action_protocol=action_protocol,
            fuel_saved_proxy_kg=fuel_saved
        )
