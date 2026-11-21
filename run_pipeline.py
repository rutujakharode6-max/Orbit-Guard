"""
End-to-End Pipeline Demonstration & Evaluation
Team: ECHELON (Adrutha Leelalakshmi, Rutuja Kharode, Aditi Sahu)
Project: Uncertainty-Aware Conjunction Risk Assessment: Reducing False Alarms in LEO Collision Screening from Public GP/TLE Data
"""

import os
import sys
from datetime import datetime, timezone, timedelta
import numpy as np

# Suppress Windows joblib warning
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from echelon_orbit.tle_fetcher import TLEFetcher
from echelon_orbit.sgp4_engine import SGP4Engine
from echelon_orbit.conjunction import ConjunctionAssessment
from echelon_orbit.triage import ConjunctionTriage
from echelon_orbit.uncertainty_model import UncertaintyModel

def run_benchmarks():
    print("=" * 80)
    print(" ECHELON ORBIT GUARD: UNCERTAINTY-AWARE CONJUNCTION RISK ASSESSMENT")
    print(" Powered by Data-Driven Quantile Regression & SGP4 Astrodynamics")
    print("=" * 80)

    # Initialize model and assessment engine
    unc_model = UncertaintyModel()
    print(f"\n[+] Loaded Uncertainty Model: {'ML-Trained HistGBRT Quantile Regressor' if unc_model.is_ml_loaded else 'Analytical Physics Fallback'}")
    
    benchmarks = TLEFetcher.load_benchmarks()
    screener = ConjunctionAssessment(uncertainty_model=unc_model)

    print(f"[+] Loaded {len(benchmarks)} Multi-Regime Operational Conjunction Scenarios.\n")

    results_summary = []

    for idx, bench in enumerate(benchmarks, 1):
        print("-" * 80)
        print(f"CASE {idx}: {bench['name']}")
        print(f"Regime: {bench['regime']}")
        print(f"Space Weather Conditions: F10.7 = {bench['space_weather']['f107']} sfu | Kp = {bench['space_weather']['kp']}")
        
        # Instantiate primary and secondary SGP4 engines
        p_eng = SGP4Engine(bench['primary']['line1'], bench['primary']['line2'], bench['primary']['name'])
        s_eng = SGP4Engine(bench['secondary']['line1'], bench['secondary']['line2'], bench['secondary']['name'])

        # Numerical TCA search: Find the exact physical close approach timestamp
        start_time = p_eng.epoch_utc
        search_window = start_time + timedelta(hours=2.5)
        
        tca_utc, calculated_miss = screener.find_tca(
            engine1=p_eng,
            engine2=s_eng,
            start_time=start_time,
            end_time=search_window,
            coarse_step_seconds=10.0
        )

        dt_hours = float(bench['delta_t_hours'])

        # Evaluate true conjunction with zero hardcoded overrides
        conjunction_res = screener.evaluate_conjunction(
            engine1=p_eng,
            engine2=s_eng,
            tca_utc=tca_utc,
            hard_body_radius1_m=bench['primary']['radius_m'],
            hard_body_radius2_m=bench['secondary']['radius_m'],
            f107=bench['space_weather']['f107'],
            kp=bench['space_weather']['kp'],
            is_maneuvering1=bench['primary']['maneuverable'],
            is_maneuvering2=bench['secondary']['maneuverable'],
            delta_t_hours=dt_hours
        )

        # Operational decision
        decision = ConjunctionTriage.evaluate(conjunction_res)
        results_summary.append({
            "case": idx,
            "name": bench['name'],
            "regime": bench['regime'],
            "naive_status": bench['naive_status'],
            "triage": decision,
            "res": conjunction_res
        })

        # Display transparent technical metrics
        sr_m = conjunction_res.cov1_rtn[0, 0] ** 0.5 * 1000.0
        st_m = conjunction_res.cov1_rtn[1, 1] ** 0.5 * 1000.0
        sn_m = conjunction_res.cov1_rtn[2, 2] ** 0.5 * 1000.0

        print(f"\n  [SGP4 Continuous Numerical Optimization Output]")
        print(f"  * Exact Computed TCA:        {tca_utc.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} UTC")
        print(f"  * Physical Miss Distance:    {conjunction_res.miss_distance_km * 1000.0:,.1f} meters")
        print(f"  * Relative Velocity at TCA:  {conjunction_res.relative_velocity_km_s:.3f} km/s")
        print(f"  * Combined Hard-Body Radius: {conjunction_res.hard_body_radius_m:.2f} meters")
        
        print(f"\n  [Learned RTN Covariance (1-Sigma Bounds)]")
        print(f"  * Radial (R):      {sr_m:.1f} m")
        print(f"  * Along-Track (T): {st_m:,.1f} m {'(Drag Expanded)' if st_m > 500 else ''}")
        print(f"  * Cross-Track (N): {sn_m:.1f} m")

        print(f"\n  [2D Encounter Plane (B-Plane) Pc & Triage Decision]")
        print(f"  * Nominal Collision Prob (Pc): {decision.nominal_pc:.3e}")
        print(f"  * Dilution Peak Pc (Pc,max):   {decision.max_pc:.3e} (Diluted: {decision.is_diluted})")
        print(f"  * Operational Action:          [{decision.action}]")
        print(f"  * Technical Rationale:         {decision.rationale}")
        print(f"  * Standard Protocol:           {decision.action_protocol}")

    # Dynamic summary computation
    total_cases = len(results_summary)
    downgraded = sum(1 for r in results_summary if r["triage"].action == "IGNORE")
    monitored = sum(1 for r in results_summary if r["triage"].action == "MONITOR")
    action_req = sum(1 for r in results_summary if r["triage"].action == "ACT")
    dilution_detected = sum(1 for r in results_summary if r["triage"].is_diluted)
    fuel_saved = sum(r["triage"].fuel_saved_proxy_kg for r in results_summary)

    print("\n" + "=" * 80)
    print(" OPERATIONAL TRIAGE & IMPACT METRICS (DYNAMICALLY COMPUTED)")
    print("=" * 80)
    print(f"• Total Conjunction Events Evaluated:  {total_cases}")
    print(f"• False Alarm Downgrade Rate:          {downgraded / total_cases * 100:.1f}% ({downgraded}/{total_cases} benign passes safely dismissed)")
    print(f"• Monitoring Escalations:              {monitored / total_cases * 100:.1f}% ({monitored}/{total_cases} queued for high-priority tracking)")
    print(f"• Mandatory Action / Maneuvers (CAM):  {action_req / total_cases * 100:.1f}% ({action_req}/{total_cases} verified high-risk alerts)")
    print(f"• Dilution Traps Identified:           {dilution_detected} (Prevented false dismissal of high-uncertainty event)")
    print(f"• Estimated Fleet Propellant Saved:    ~{fuel_saved:.2f} kg hydrazine proxy per screening cycle")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    run_benchmarks()
