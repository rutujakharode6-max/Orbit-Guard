"""
ECHELON Orbit Guard: Streamlit Decision Dashboard
Uncertainty-Aware Conjunction Risk Assessment in LEO
Team: ECHELON (Adrutha Leelalakshmi, Rutuja Kharode, Aditi Sahu)
Theme: Sustainable Space and Space Environment
"""

import os
import sys
import math
import json
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Suppress Windows joblib warning
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

# Configure page layout
st.set_page_config(
    page_title="ECHELON | Orbit Guard",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Project path resolution
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from echelon_orbit.tle_fetcher import TLEFetcher
from echelon_orbit.sgp4_engine import SGP4Engine, EARTH_RADIUS_KM
from echelon_orbit.frames import teme_to_rtn
from echelon_orbit.uncertainty_model import UncertaintyModel
from echelon_orbit.conjunction import ConjunctionAssessment, ConjunctionResult
from echelon_orbit.collision_prob import CollisionProbability
from echelon_orbit.triage import ConjunctionTriage
from echelon_orbit.error_analyzer import ErrorAnalyzer

# Styling
st.markdown("""
<style>
    .reportview-container {
        background: #0d1117;
    }
    .main-header {
        font-family: 'Inter', -apple-system, sans-serif;
        background: linear-gradient(90deg, #1f2937, #111827);
        padding: 24px;
        border-radius: 12px;
        border-left: 6px solid #3b82f6;
        margin-bottom: 25px;
    }
    .metric-card {
        background: #1e293b;
        padding: 18px;
        border-radius: 10px;
        border: 1px solid #334155;
        text-align: center;
    }
    .metric-val {
        font-size: 24px;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-lbl {
        font-size: 13px;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .badge-act {
        background-color: #ef4444;
        color: white;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
    }
    .badge-monitor {
        background-color: #f59e0b;
        color: white;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
    }
    .badge-ignore {
        background-color: #10b981;
        color: white;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# Application Header
st.markdown("""
<div class="main-header">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <h1 style="margin: 0; color: #f8fafc; font-size: 28px;">🛰️ ECHELON Orbit Guard</h1>
            <p style="margin: 5px 0 0 0; color: #94a3b8; font-size: 15px;">
                <strong>Uncertainty-Aware Conjunction Risk Assessment:</strong> Reducing False Alarms in LEO Collision Screening from Public GP/TLE Data
            </p>
        </div>
        <div style="text-align: right;">
            <span style="background: #2563eb; color: white; padding: 4px 10px; border-radius: 6px; font-size: 12px; font-weight: bold;">TEAM ECHELON</span>
            <p style="margin: 4px 0 0 0; color: #cbd5e1; font-size: 12px;">Adrutha L. · Rutuja K. · Aditi S.</p>
            <p style="margin: 0; color: #64748b; font-size: 11px;">Sustainable Space Environment</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Data & Model Initialization
catalog = TLEFetcher.load_catalog()
benchmarks = TLEFetcher.load_benchmarks()
space_weather_data = TLEFetcher.load_space_weather()
unc_model = UncertaintyModel()
conjunction_screener = ConjunctionAssessment(unc_model)
error_analyzer = ErrorAnalyzer(unc_model)

# Load trained metrics if available
metrics_file = os.path.join(PROJECT_ROOT, "data", "model_metrics.json")
ml_metrics = None
if os.path.exists(metrics_file):
    try:
        with open(metrics_file, "r", encoding="utf-8") as f:
            ml_metrics = json.load(f)
    except Exception:
        pass

# Sidebar Navigation
st.sidebar.title("Navigation & Controls")
page = st.sidebar.radio(
    "Select Workflow Module:",
    [
        "⚡ Conjunction Assessment & Triage",
        "🎯 Benchmark Scenarios Replay",
        "🤖 Trained Quantile ML Models",
        "📈 Empirical RTN Uncertainty Growth",
        "🌀 Encounter Plane (B-Plane) & Dilution",
        "🌐 3D Orbit Trajectory Viewer",
        "📚 Research & Scientific Foundations"
    ]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Global Space Weather Condition")
f107_slider = st.sidebar.slider("Solar Radio Flux F10.7 (sfu)", 70.0, 260.0, 185.0, help="Solar EUV radiation driving upper thermosphere expansion.")
kp_slider = st.sidebar.slider("Geomagnetic Planetary Index Kp", 0.0, 9.0, 3.5, help="Geomagnetic activity index driving storm drag.")

st.sidebar.markdown("---")
st.sidebar.markdown(f"""
<div style="font-size: 12px; color: #94a3b8;">
<strong>Uncertainty Engine:</strong><br>
• Model: <span style="color: {'#38bdf8' if unc_model.is_ml_loaded else '#f59e0b'};">{'HistGBRT Quantile Regressor' if unc_model.is_ml_loaded else 'Physics Fallback'}</span><br>
• Objects in Catalog: <strong>{len(catalog)}</strong><br>
• SGP4 Frame: <strong>TEME</strong><br>
• Covariance: <strong>Coupled RTN (ρ_RT=0.30)</strong><br>
• Encounter: <strong>Foster 2D Integral</strong>
</div>
""", unsafe_allow_html=True)

# ==========================================
# PAGE 1: CONJUNCTION ASSESSMENT & TRIAGE
# ==========================================
if page == "⚡ Conjunction Assessment & Triage":
    st.subheader("Pairwise Conjunction Risk Screening")
    
    # Mode Toggle: Explicitly distinguishing Real SGP4 Search vs What-If Sandbox
    eval_mode = st.radio(
        "Evaluation Mode:",
        ["🔬 Mode A: Real Propagated Trajectory Screening (Zero Overrides)", "🎛️ Mode B: What-If Sensitivity & Stress-Test Sandbox"],
        horizontal=True
    )

    col_p, col_s = st.columns(2)

    with col_p:
        st.markdown("#### Primary Asset (Protected Satellite)")
        primary_names = [f"{o['norad_id']} - {o['name']} ({o['type']})" for o in catalog]
        iss_idx = next((i for i, o in enumerate(catalog) if o['norad_id'] == 25544), 0)
        p_sel = st.selectbox("Select Primary Satellite:", primary_names, index=iss_idx)
        p_norad = int(p_sel.split(" - ")[0])
        p_obj = TLEFetcher.get_by_norad_id(p_norad)
        
        p_hbr = st.number_input("Primary Radius / HBR (m):", min_value=0.1, max_value=100.0, value=float(p_obj.get("hard_body_radius_m", 2.0)))
        p_maneuver = st.checkbox("Has Propulsion / Maneuverable", value=p_obj.get("maneuverable", False))

    with col_s:
        st.markdown("#### Secondary Object (Debris / Interceptor)")
        deb_names = [f"{o['norad_id']} - {o['name']} ({o['type']})" for o in catalog if o['norad_id'] != p_norad]
        s_sel = st.selectbox("Select Secondary Object:", deb_names, index=min(12, len(deb_names)-1))
        s_norad = int(s_sel.split(" - ")[0])
        s_obj = TLEFetcher.get_by_norad_id(s_norad)
        
        s_hbr = st.number_input("Secondary Radius / HBR (m):", min_value=0.05, max_value=20.0, value=float(s_obj.get("hard_body_radius_m", 0.35)))
        s_maneuver = st.checkbox("Secondary Maneuverable", value=s_obj.get("maneuverable", False), disabled=True)

    p_engine = SGP4Engine(p_obj['line1'], p_obj['line2'], p_obj['name'])
    s_engine = SGP4Engine(s_obj['line1'], s_obj['line2'], s_obj['name'])

    st.markdown("---")

    if "Mode A" in eval_mode:
        st.info("ℹ️ **Mode A (Real Trajectory Mode):** SGP4 propagates both orbits across a search window, continuously finds the exact mathematical Time of Closest Approach (TCA) via Brent numerical minimization, computes the exact geometric miss distance, and predicts ML-calibrated covariance.")
        
        c_search1, c_search2 = st.columns(2)
        with c_search1:
            search_window_hours = st.slider("Search Window Horizon (hours):", 6.0, 72.0, 24.0)
        with c_search2:
            step_sec = st.select_slider("Coarse Scan Step (seconds):", options=[15.0, 30.0, 60.0, 120.0], value=30.0)

        with st.spinner("Propagating orbits and performing continuous numerical minimization for TCA..."):
            start_t = p_engine.epoch_utc
            end_t = start_t + timedelta(hours=search_window_hours)
            
            tca_found, real_min_dist = conjunction_screener.find_tca(
                engine1=p_engine,
                engine2=s_engine,
                start_time=start_t,
                end_time=end_t,
                coarse_step_seconds=step_sec
            )

            dt_propagation = abs((tca_found - p_engine.epoch_utc).total_seconds()) / 3600.0

            conj_res = conjunction_screener.evaluate_conjunction(
                engine1=p_engine,
                engine2=s_engine,
                tca_utc=tca_found,
                hard_body_radius1_m=p_hbr,
                hard_body_radius2_m=s_hbr,
                f107=f107_slider,
                kp=kp_slider,
                is_maneuvering1=p_maneuver,
                is_maneuvering2=s_maneuver,
                delta_t_hours=dt_propagation
            )

            triage_dec = ConjunctionTriage.evaluate(conj_res)

        st.success(f"✓ Found True TCA at **{tca_found.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} UTC** (+{dt_propagation:.2f} hrs from epoch)")

    else:
        st.warning("⚠️ **Mode B (What-If Sandbox):** Stress-test operational triage across hypothetical miss distances and propagation delays.")
        cs1, cs2 = st.columns(2)
        with cs1:
            whatif_dt = st.slider("Hypothetical Prediction Horizon Δt (hours):", 2.0, 96.0, 24.0)
        with cs2:
            whatif_miss_m = st.slider("Hypothetical Miss Distance (meters):", 20.0, 3000.0, 480.0)

        eval_tca = p_engine.epoch_utc + timedelta(hours=whatif_dt)
        conj_res = conjunction_screener.evaluate_conjunction(
            engine1=p_engine,
            engine2=s_engine,
            tca_utc=eval_tca,
            hard_body_radius1_m=p_hbr,
            hard_body_radius2_m=s_hbr,
            f107=f107_slider,
            kp=kp_slider,
            is_maneuvering1=p_maneuver,
            is_maneuvering2=s_maneuver,
            delta_t_hours=whatif_dt
        )
        
        # Override only in sandbox mode
        miss_km = whatif_miss_m / 1000.0
        conj_res.miss_distance_km = miss_km
        conj_res.miss_2d_encounter = np.array([miss_km * 0.85, miss_km * 0.52])
        triage_dec = ConjunctionTriage.evaluate(conj_res)

    # Display Triage Decision Banner
    st.markdown("### Operational Conjunction Triage")
    badge_class = f"badge-{triage_dec.action.lower()}"
    
    st.markdown(f"""
    <div style="background: #1e293b; padding: 20px; border-radius: 12px; border-left: 8px solid {triage_dec.status_color}; margin-bottom: 20px;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <span class="{badge_class}" style="font-size: 20px;">ACTION: {triage_dec.action}</span>
                <span style="margin-left: 15px; color: #94a3b8; font-size: 15px;">Combined HBR: <strong>{triage_dec.hbr_m:.2f} m</strong></span>
            </div>
            <div style="color: #cbd5e1; font-size: 14px;">
                Calculated Miss: <strong>{conj_res.miss_distance_km * 1000.0:,.1f} meters</strong>
            </div>
        </div>
        <p style="margin: 15px 0 8px 0; color: #f1f5f9; font-size: 16px;">
            <strong>Rationale:</strong> {triage_dec.rationale}
        </p>
        <p style="margin: 0; color: #38bdf8; font-size: 14px;">
            <strong>Operator Protocol:</strong> {triage_dec.action_protocol}
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Key Metrics Columns
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-lbl">Physical Miss Distance</div>
            <div class="metric-val">{conj_res.miss_distance_km * 1000.0:,.1f} m</div>
            <div style="font-size: 11px; color: #64748b;">Relative Speed: {conj_res.relative_velocity_km_s:.2f} km/s</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-lbl">Calibrated Collision Prob (Pc)</div>
            <div class="metric-val">{triage_dec.nominal_pc:.2e}</div>
            <div style="font-size: 11px; color: #64748b;">Action threshold: ≥ 1.0e-4</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-lbl">Peak Dilution Risk (Pc,max)</div>
            <div class="metric-val">{triage_dec.max_pc:.2e}</div>
            <div style="font-size: 11px; color: {'#f59e0b' if triage_dec.is_diluted else '#10b981'};">
                {'⚠️ Dilution Trap Detected' if triage_dec.is_diluted else '✓ Not In Dilution Region'}
            </div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        sr = conj_res.cov1_rtn[0,0]**0.5 * 1000.0
        st_val = conj_res.cov1_rtn[1,1]**0.5 * 1000.0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-lbl">Learned 1σ Bounds</div>
            <div class="metric-val">{st_val:,.0f} m</div>
            <div style="font-size: 11px; color: #64748b;">Radial: {sr:.0f} m | In-Track: {st_val:,.0f} m</div>
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# PAGE 2: BENCHMARK SCENARIOS REPLAY
# ==========================================
elif page == "🎯 Benchmark Scenarios Replay":
    st.subheader("Hackathon Benchmark Replay: Validating Triage across 4 LEO Regimes")
    st.markdown("""
    Evaluated with **continuous numerical TCA minimization** and **ML quantile regression**.
    Notice how calibrated uncertainty separates benign alarms from genuine collision threats and dilution traps.
    """)

    for idx, bench in enumerate(benchmarks, 1):
        with st.expander(f"Case {idx}: {bench['name']} ({bench['regime']})", expanded=(idx <= 2)):
            col_b1, col_b2 = st.columns([3, 2])
            
            with col_b1:
                st.markdown(f"**Baseline Status:** `{bench['naive_status']}`")
                st.markdown(f"**Expected Close Pass:** `~{bench.get('expected_miss_km', 0.5)*1000:.0f} meters` | **Prediction Arc:** `{bench['delta_t_hours']} hours`")
                st.markdown(f"**Space Weather:** Solar Flux $F_{{10.7}} = {bench['space_weather']['f107']}$ sfu, Geomagnetic Index $K_p = {bench['space_weather']['kp']}$")
                st.info(bench["narrative"])

            with col_b2:
                # Real propagation & TCA evaluation
                p_eng = SGP4Engine(bench['primary']['line1'], bench['primary']['line2'], bench['primary']['name'])
                s_eng = SGP4Engine(bench['secondary']['line1'], bench['secondary']['line2'], bench['secondary']['name'])
                
                tca_opt, _ = conjunction_screener.find_tca(
                    engine1=p_eng,
                    engine2=s_eng,
                    start_time=p_eng.epoch_utc,
                    end_time=p_eng.epoch_utc + timedelta(hours=2.5),
                    coarse_step_seconds=10.0
                )

                c_res = conjunction_screener.evaluate_conjunction(
                    engine1=p_eng,
                    engine2=s_eng,
                    tca_utc=tca_opt,
                    hard_body_radius1_m=bench['primary']['radius_m'],
                    hard_body_radius2_m=bench['secondary']['radius_m'],
                    f107=bench['space_weather']['f107'],
                    kp=bench['space_weather']['kp'],
                    is_maneuvering1=bench['primary']['maneuverable'],
                    is_maneuvering2=bench['secondary']['maneuverable'],
                    delta_t_hours=bench['delta_t_hours']
                )
                dec = ConjunctionTriage.evaluate(c_res)

                st.markdown(f"""
                <div style="background: #0f172a; padding: 14px; border-radius: 8px; border: 1px solid #334155;">
                    <div style="font-size: 13px; color: #94a3b8;">CALCULATED TCA & TRIAGE:</div>
                    <div style="font-size: 20px; font-weight: bold; color: {dec.status_color};">DECISION: {dec.action}</div>
                    <div style="margin-top: 6px; font-size: 13px; color: #e2e8f0;">
                        Computed TCA: <strong>{tca_opt.strftime('%H:%M:%S.%f')[:-3]} UTC</strong><br>
                        Propagated Miss: <strong>{c_res.miss_distance_km * 1000.0:,.1f} m</strong><br>
                        Nominal Pc: <strong>{dec.nominal_pc:.2e}</strong><br>
                        Max Dilution Pc: <strong>{dec.max_pc:.2e}</strong><br>
                        Along-Track 1σ: <strong>{c_res.cov1_rtn[1,1]**0.5*1000:,.0f} m</strong>
                    </div>
                </div>
                """, unsafe_allow_html=True)

# ==========================================
# PAGE 3: TRAINED QUANTILE ML MODELS
# ==========================================
elif page == "🤖 Trained Quantile ML Models":
    st.subheader("Data-Driven Quantile Regression Error Models")
    st.markdown("""
    Trained on **3,250 empirical SGP4-vs-reference residual arcs** across the 4 LEO regimes using `HistGradientBoostingRegressor(loss="quantile")`.
    Predicts data-driven uncertainty bounds directly from orbital parameters ($B^*$, altitude, eccentricity, inclination) and space weather ($F_{10.7}, K_p$).
    """)

    if ml_metrics is not None:
        st.markdown("#### Test Set Empirical Calibration Coverage (Held-Out 20% Split)")
        
        col_q1, col_q2, col_q3 = st.columns(3)
        with col_q1:
            rad_1s = ml_metrics["axes"]["radial"]["1_sigma"]["empirical_coverage_pct"]
            rad_2s = ml_metrics["axes"]["radial"]["2_sigma"]["empirical_coverage_pct"]
            st.metric("Radial (R) 1σ Coverage", f"{rad_1s}%", f"Target: 84.1%")
            st.caption(f"2σ Bound Coverage: **{rad_2s}%** (Target: 95.5%) | MAE: {ml_metrics['axes']['radial']['1_sigma']['mae_km']*1000:.1f} m")

        with col_q2:
            in_1s = ml_metrics["axes"]["intrack"]["1_sigma"]["empirical_coverage_pct"]
            in_2s = ml_metrics["axes"]["intrack"]["2_sigma"]["empirical_coverage_pct"]
            st.metric("Along-Track (T) 1σ Coverage", f"{in_1s}%", f"Target: 84.1%")
            st.caption(f"2σ Bound Coverage: **{in_2s}%** (Target: 95.5%) | MAE: {ml_metrics['axes']['intrack']['1_sigma']['mae_km']*1000:.1f} m")

        with col_q3:
            cr_1s = ml_metrics["axes"]["crosstrack"]["1_sigma"]["empirical_coverage_pct"]
            cr_2s = ml_metrics["axes"]["crosstrack"]["2_sigma"]["empirical_coverage_pct"]
            st.metric("Cross-Track (N) 1σ Coverage", f"{cr_1s}%", f"Target: 84.1%")
            st.caption(f"2σ Bound Coverage: **{cr_2s}%** (Target: 95.5%) | MAE: {ml_metrics['axes']['crosstrack']['1_sigma']['mae_km']*1000:.1f} m")

        st.markdown("---")
        st.markdown("#### Model Architecture & Features")
        st.json({
            "Algorithm": "Histogram Gradient Boosted Quantile Decision Trees (HistGBRT)",
            "Features Used": ml_metrics["features"],
            "Quantile Levels Trained": [0.50, 0.8413, 0.9545, 0.9973],
            "Training Records": ml_metrics["train_samples"],
            "Validation Records": ml_metrics["test_samples"]
        })
    else:
        st.warning("Training metrics file not found. Run `python train_quantile_model.py` to regenerate.")

# ==========================================
# PAGE 4: EMPIRICAL RTN ERROR & UNCERTAINTY
# ==========================================
elif page == "📈 Empirical RTN Uncertainty Growth":
    st.subheader("Empirical SGP4 Error Growth Modeling in RTN Coordinates")
    st.markdown(r"""
    SGP4 position error grows non-linearly over propagation time $\Delta t$. While radial and cross-track
    errors remain tightly bounded, along-track error expands quadratically due to atmospheric drag variability.
    """)

    sample_sat = catalog[0]
    sat_eng = SGP4Engine(sample_sat['line1'], sample_sat['line2'], sample_sat['name'])

    alt_choice = st.slider("Mean Orbit Altitude (km):", 300.0, 900.0, float(sat_eng.elements.mean_altitude_km))
    bstar_choice = st.slider("B* Drag Term (1/Earth radii):", 0.00001, 0.00200, float(max(1e-5, abs(sat_eng.elements.bstar))), format="%.5f")

    dt_range = np.linspace(1.0, 72.0, 50)
    sig_r_arr, sig_t_arr, sig_n_arr = [], [], []

    for dt in dt_range:
        sr, st_val, sn = unc_model.predict_rtn_sigmas(
            delta_t_hours=dt,
            altitude_km=alt_choice,
            bstar=bstar_choice,
            f107=f107_slider,
            kp=kp_slider
        )
        sig_r_arr.append(sr * 1000.0)
        sig_t_arr.append(st_val * 1000.0)
        sig_n_arr.append(sn * 1000.0)

    fig_rtn = go.Figure()
    fig_rtn.add_trace(go.Scatter(x=dt_range, y=sig_t_arr, mode='lines', name='Along-Track / In-Track (T)', line=dict(color='#ef4444', width=3)))
    fig_rtn.add_trace(go.Scatter(x=dt_range, y=sig_n_arr, mode='lines', name='Cross-Track / Normal (N)', line=dict(color='#3b82f6', width=2, dash='dash')))
    fig_rtn.add_trace(go.Scatter(x=dt_range, y=sig_r_arr, mode='lines', name='Radial (R)', line=dict(color='#10b981', width=2, dash='dot')))

    fig_rtn.update_layout(
        title=f"1σ Uncertainty Growth vs Propagation Horizon Δt (Alt: {alt_choice:.0f} km, F10.7: {f107_slider:.0f} sfu)",
        xaxis_title="Propagation Interval Δt (hours)",
        yaxis_title="Position Uncertainty 1σ (meters)",
        template="plotly_dark",
        height=500,
        legend=dict(x=0.02, y=0.95)
    )
    st.plotly_chart(fig_rtn, use_container_width=True)

# ==========================================
# PAGE 5: ENCOUNTER PLANE & DILUTION
# ==========================================
elif page == "🌀 Encounter Plane (B-Plane) & Dilution":
    st.subheader("2D Encounter Plane (B-Plane) & The Dilution Paradox")
    st.markdown(r"""
    At Time of Closest Approach (TCA), the conjunction geometry is projected onto the 2D plane perpendicular
    to relative velocity $\mathbf{v}_{\text{rel}}$. The combined covariance $\mathbf{\Sigma}_{2D}$ forms an uncertainty ellipse.
    """)

    col_e1, col_e2 = st.columns([1, 1])

    with col_e1:
        st.markdown("#### Encounter Parameters")
        miss_x = st.slider("Encounter X Miss Distance (meters):", -1500, 1500, 350)
        miss_y = st.slider("Encounter Y Miss Distance (meters):", -1500, 1500, 220)
        sigma_major = st.slider("Major Principal Axis 1σ (meters):", 100, 2500, 650)
        sigma_minor = st.slider("Minor Principal Axis 1σ (meters):", 50, 1000, 180)
        ellipse_angle = st.slider("Covariance Ellipse Tilt (°):", 0, 180, 35)
        hbr_val = st.slider("Combined Hard-Body Radius (meters):", 1.0, 50.0, 10.0)

    # Convert to km
    miss_2d_km = np.array([miss_x / 1000.0, miss_y / 1000.0])
    hbr_km = hbr_val / 1000.0
    th_rad = math.radians(ellipse_angle)
    
    # Construct 2D covariance
    d_mat = np.array([
        [(sigma_major/1000.0)**2, 0.0],
        [0.0, (sigma_minor/1000.0)**2]
    ])
    rot = np.array([
        [math.cos(th_rad), -math.sin(th_rad)],
        [math.sin(th_rad), math.cos(th_rad)]
    ])
    cov_2d = rot @ d_mat @ rot.T

    # 1. Plot Encounter Plane
    theta = np.linspace(0, 2*np.pi, 100)
    
    def get_ellipse_pts(scale_sig):
        pts = np.vstack([scale_sig * (sigma_major) * np.cos(theta), scale_sig * (sigma_minor) * np.sin(theta)])
        rot_pts = rot @ pts
        return rot_pts[0, :], rot_pts[1, :]

    fig_enc = go.Figure()
    
    x3, y3 = get_ellipse_pts(3.0)
    fig_enc.add_trace(go.Scatter(x=x3, y=y3, mode='lines', name='3σ Uncertainty', line=dict(color='rgba(59, 130, 246, 0.3)', width=1)))
    
    x2, y2 = get_ellipse_pts(2.0)
    fig_enc.add_trace(go.Scatter(x=x2, y=y2, mode='lines', name='2σ Uncertainty', line=dict(color='rgba(59, 130, 246, 0.6)', width=2)))
    
    x1, y1 = get_ellipse_pts(1.0)
    fig_enc.add_trace(go.Scatter(x=x1, y=y1, mode='lines', name='1σ Uncertainty', line=dict(color='#3b82f6', width=2)))

    fig_enc.add_trace(go.Scatter(
        x=[miss_x], y=[miss_y],
        mode='markers+text',
        name='Nominal Miss Vector',
        marker=dict(size=12, color='#ef4444', symbol='x'),
        text=["Secondary (TCA)"], textposition="top right"
    ))

    xhbr = miss_x + hbr_val * np.cos(theta)
    yhbr = miss_y + hbr_val * np.sin(theta)
    fig_enc.add_trace(go.Scatter(
        x=xhbr, y=yhbr,
        mode='lines',
        name=f'Hard Body Disk (Rc={hbr_val}m)',
        line=dict(color='#f59e0b', width=2),
        fill='toself',
        fillcolor='rgba(245, 158, 11, 0.2)'
    ))

    fig_enc.add_trace(go.Scatter(
        x=[0], y=[0],
        mode='markers+text',
        name='Primary Asset (Origin)',
        marker=dict(size=10, color='#10b981', symbol='circle'),
        text=["Primary"], textposition="bottom left"
    ))

    fig_enc.update_layout(
        title="2D Encounter Plane (B-Plane) Geometry",
        xaxis_title="Encounter Plane ξ (meters)",
        yaxis_title="Encounter Plane ζ (meters)",
        template="plotly_dark",
        height=450,
        yaxis=dict(scaleanchor="x", scaleratio=1)
    )

    with col_e2:
        st.plotly_chart(fig_enc, use_container_width=True)

    # 2. Dilution Curve
    st.markdown(r"#### Dilution Curve: Collision Probability $P_c$ vs. Uncertainty Scale $\lambda$")
    dilution_res = CollisionProbability.analyze_dilution(miss_2d_km, cov_2d, hbr_km)
    
    dil_df = pd.DataFrame(dilution_res["curve_data"])
    fig_dil = go.Figure()
    fig_dil.add_trace(go.Scatter(x=dil_df["scale"], y=dil_df["pc"], mode='lines', name='Pc(λ)', line=dict(color='#38bdf8', width=3)))
    
    fig_dil.add_trace(go.Scatter(
        x=[1.0], y=[dilution_res["nominal_pc"]],
        mode='markers+text',
        name='Nominal Operational Scale (λ=1.0)',
        marker=dict(size=12, color='#ef4444'),
        text=[f"Nominal: {dilution_res['nominal_pc']:.2e}"], textposition="top right"
    ))

    fig_dil.update_layout(
        title="Dilution Effect: Notice how large uncertainty paradoxically drops nominal Pc",
        xaxis_title="Uncertainty Scale Factor λ (Covariance = λ² · Cov_nominal)",
        yaxis_title="Collision Probability Pc",
        xaxis_type="log",
        yaxis_type="log",
        template="plotly_dark",
        height=380
    )
    st.plotly_chart(fig_dil, use_container_width=True)

# ==========================================
# PAGE 6: 3D ORBIT TRAJECTORY VIEWER
# ==========================================
elif page == "🌐 3D Orbit Trajectory Viewer":
    st.subheader("3D Interactive Orbit Visualizer (TEME Frame)")
    st.markdown("Interactive 3D orbital propagation showing relative motion and encounter geometry.")

    sat1 = catalog[0] # ISS
    sat2 = next((o for o in catalog if "COSMOS" in o["name"]), catalog[5])

    eng1 = SGP4Engine(sat1['line1'], sat1['line2'], sat1['name'])
    eng2 = SGP4Engine(sat2['line1'], sat2['line2'], sat2['name'])

    traj1, _, _ = eng1.generate_orbit_trajectory(eng1.epoch_utc, duration_minutes=95.0, step_seconds=45.0)
    traj2, _, _ = eng2.generate_orbit_trajectory(eng1.epoch_utc, duration_minutes=95.0, step_seconds=45.0)

    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 40)
    x_earth = EARTH_RADIUS_KM * np.outer(np.cos(u), np.sin(v))
    y_earth = EARTH_RADIUS_KM * np.outer(np.sin(u), np.sin(v))
    z_earth = EARTH_RADIUS_KM * np.outer(np.ones(np.size(u)), np.cos(v))

    fig_3d = go.Figure()
    fig_3d.add_trace(go.Surface(x=x_earth, y=y_earth, z=z_earth, colorscale='Blues', showscale=False, opacity=0.85, name="Earth"))

    fig_3d.add_trace(go.Scatter3d(x=traj1[:, 0], y=traj1[:, 1], z=traj1[:, 2], mode='lines', name=eng1.name, line=dict(color='#10b981', width=4)))
    fig_3d.add_trace(go.Scatter3d(x=traj2[:, 0], y=traj2[:, 1], z=traj2[:, 2], mode='lines', name=eng2.name, line=dict(color='#ef4444', width=3)))

    fig_3d.update_layout(
        title="3D Earth & LEO Trajectories",
        scene=dict(
            xaxis_title="TEME X (km)",
            yaxis_title="TEME Y (km)",
            zaxis_title="TEME Z (km)",
            aspectmode='data'
        ),
        template="plotly_dark",
        height=650
    )
    st.plotly_chart(fig_3d, use_container_width=True)

# ==========================================
# PAGE 7: RESEARCH & SCIENTIFIC FOUNDATIONS
# ==========================================
elif page == "📚 Research & Scientific Foundations":
    st.subheader("Astrodynamic Foundations & Verified Literature References")
    
    st.markdown("""
    ### Key Theoretical Pillars
    
    1. **Flohrer et al. (AMOS 2008) & Oltrogge et al. (AMOS 2014):**
       - Established that TLE error is predominantly along-track due to atmospheric drag mismodeling.
       - Demonstrated empirical residual binning in Radial, In-Track, and Cross-Track (RTN/RSW) coordinates to derive calibrated covariance bounds.
       
    2. **NASA Conjunction Assessment & Covariance Manipulation (NASA NTRS 2016 / 2019):**
       - Formalized the 2D encounter plane projection perpendicular to relative velocity.
       - Solved the **Dilution Paradox**: when state uncertainty is excessively large, probability density spreads over space, causing raw $P_c$ to drop artificially.
       
    3. **Foster & Alfano Formulation (AAS / AIAA):**
       - Closed-form and rotated principal axis integration of the bivariate Gaussian probability density function over the combined hard-body radius $R_c$.
    
    ---
    ### Academic Citations
    - Flohrer, T., et al. *"Assessment and Categorization of TLE Orbit Errors for the US SSN."* Advanced Maui Optical and Space Surveillance Technologies (AMOS), 2008.
    - Oltrogge, D. L., et al. *"Parametric Characterization of SGP4 Theory and TLE Positional Errors."* AMOS Conference, 2014.
    - NASA Goddard Space Flight Center. *"Covariance Manipulation for Conjunction Assessment."* NASA NTRS 20160010497.
    - Embry-Riddle Aeronautical University. *"Satellite Conjunction Assessment Risk Analysis for Dilution Region."* Space Traffic Management Conference.
    """)
