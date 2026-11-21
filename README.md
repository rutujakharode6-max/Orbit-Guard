# ECHELON Orbit Guard: Uncertainty-Aware Conjunction Risk Assessment

> **Theme:** Sustainable Space and Space Environment  
> **Team Name:** ECHELON  
> **Team Members:** Adrutha Leelalakshmi · Rutuja Kharode · Aditi Sahu  

---

## 🎯 Executive Summary & Problem Statement

### Title
**Uncertainty-Aware Conjunction Risk Assessment: Reducing False Alarms in LEO Collision Screening from Public GP/TLE Data**

### The Core Problem
Low Earth Orbit (LEO) hosts tens of thousands of cataloged objects. Operators must evaluate close-approach alerts and decide whether to execute collision-avoidance maneuvers. 

Standard screening based strictly on raw miss distance (e.g. flagging everything under 1 km or 5 km) on SGP4-propagated states creates **extremely high false-positive rates**:
- SGP4/GP data omits full covariance matrices and suffers from kilometer-scale positional errors that grow rapidly with propagation interval ($\Delta t$) and space-weather-driven atmospheric drag variability.
- Predicted close approaches at sub-kilometer miss distances are frequently indistinguishable from safe encounters once realistic state uncertainty is accounted for.
- **Consequences:** Unnecessary maneuvers expend irreplaceable propellant, shorten mission lifespan, disrupt observations, and trigger operator alert fatigue.

---

## 🏗️ System Architecture & What Makes This Defensible

Unlike prototypes that use static placeholder tables or synthetic overrides, **ECHELON is a fully data-driven, physically computed engine**:

```
[130+ Curated LEO Objects] + [NOAA Space Weather (F10.7, Kp)]
               │
               ▼
[Multi-Arc SGP4 Residual Extraction (build_error_dataset.py)]
  ├── 3,250 empirical arcs across VLEO, Shell Debris, Sun-Sync, R/Bs
  └── Position error vector decomposed into RTN frame (Radial, In-Track, Cross-Track)
               │
               ▼
[Quantile Regression Training (train_quantile_model.py)]
  ├── HistGradientBoostingRegressor (loss="quantile", q=0.50, 0.84, 0.95, 0.99)
  ├── Features: Δt, altitude, perigee, eccentricity, inclination, B*, F10.7, Kp, maneuver_flag
  └── Validated Empirical Coverage on 20% Held-Out Test Set:
        • Radial (1σ):      81.5% (Target: 84.1%)
        • Along-Track (1σ): 79.5% (Target: 84.1%)
        • Cross-Track (1σ): 79.7% (Target: 84.1%)
               │
               ▼
[Astrodynamic Conjunction Screener (conjunction.py)]
  ├── Continuous Brent numerical minimization for true Time of Closest Approach (TCA)
  ├── SGP4 state propagation in TEME frame (Zero synthetic overrides)
  ├── Coupled RTN covariance (energy-mean motion cross-correlation ρ_RT = 0.30)
  └── 2D Encounter Plane (B-Plane) projection normal to relative velocity v_rel
               │
               ▼
[Foster/Alfano Pc & Dilution Analysis (collision_prob.py)]
  ├── Principal-axis rotated disk integration over combined hard-body radius Rc
  └── Dilution Paradox detection: sweeps scale λ to catch high-uncertainty false dismissal
               │
               ▼
[Actionable 3-Tier Triage (triage.py)]
  ├── 🟢 IGNORE   (Pc < 1e-6): Safely dismiss benign alarms, saving fuel
  ├── 🟡 MONITOR  (1e-6 <= Pc < 1e-4 or Dilution Trap): Queue high-priority tracking
  └── 🔴 ACT      (Pc >= 1e-4): Execute Collision Avoidance Maneuver (CAM)
```

---

## 🛰️ Curated Multi-Regime LEO Catalog

The catalog includes **130+ pre-staged real objects** across 4 key LEO regimes:

| Regime | Altitude Band | Representative Objects | Error Characteristic |
| :--- | :--- | :--- | :--- |
| **Regime 1: High-Drag / VLEO** | ~380–450 km | ISS (ZARYA - 25544), Tiangong CSS (48274), Planet Dove (Flock), Spire Lemur CubeSats | Rapid along-track quadratic error expansion driven by solar flux $F_{10.7}$ and thermospheric heating. |
| **Regime 2: Mega-Constellation Shell Debris** | ~500–600 km | Starlink satellites, Cosmos 2251 Debris (15+ pieces), Fengyun-1C ASAT Debris (15+ pieces), Iridium 33 Debris | Dense traffic zone. Non-maneuvering debris provides the cleanest ground-truth error arcs without thrust jumps. |
| **Regime 3: Mid-LEO Sun-Sync & Weather** | ~650–850 km | NOAA-15 (25338), NOAA-18 (28654), NOAA-19 (33591), MetOp, Sentinel-1A/2A | Low drag; errors grow slowly and predictably. Naive sub-km alarms are almost entirely benign. |
| **Regime 4: Decaying & Spent Rocket Bodies** | Decaying / Elliptical | SL-16 Upper Stages (22803), Falcon 9 R/B | Large Radar Cross-Section (RCS) and high $B^*$ drag create massive along-track uncertainty (> 5 km), triggering the Dilution Paradox. |

---

## 🧮 Mathematical & Physical Formulation

### 1. Frame Transformations (TEME $\to$ RTN $\to$ Encounter Plane)
- SGP4 propagates in native **TEME** (True Equator, Mean Equinox).
- Covariance estimation is evaluated in the **RTN / RSW** triad:
  $$\hat{R} = \frac{\mathbf{r}}{\|\mathbf{r}\|}, \quad \hat{N} = \frac{\mathbf{r} \times \mathbf{v}}{\|\mathbf{r} \times \mathbf{v}\|}, \quad \hat{T} = \hat{N} \times \hat{R}$$
  $$\mathbf{R}_{\text{TEME}\to\text{RTN}} = \begin{bmatrix} \hat{R}^T \\ \hat{T}^T \\ \hat{N}^T \end{bmatrix}, \quad \mathbf{\Sigma}_{\text{TEME}} = \mathbf{R}_{\text{TEME}\to\text{RTN}}^T \mathbf{\Sigma}_{\text{RTN}} \mathbf{R}_{\text{TEME}\to\text{RTN}}$$
- At TCA, relative velocity $\mathbf{v}_{\text{rel}} = \mathbf{v}_1 - \mathbf{v}_2$ defines the encounter plane normal. Projection matrix $\mathbf{P}_{\text{enc}} = [\hat{\xi}, \hat{\zeta}]^T$ projects combined Cartesian covariance:
  $$\mathbf{\Sigma}_{2D} = \mathbf{P}_{\text{enc}} (\mathbf{\Sigma}_1 + \mathbf{\Sigma}_2) \mathbf{P}_{\text{enc}}^T$$

### 2. Coupled RTN Covariance
Because semi-major axis perturbation couples with mean motion and in-track timing, orbital mechanics dictates a non-zero radial-in-track cross-correlation ($\rho_{RT} \approx 0.30$):
$$\mathbf{\Sigma}_{\text{RTN}} = \begin{bmatrix} \sigma_R^2 & \rho_{RT}\sigma_R\sigma_T & 0 \\ \rho_{RT}\sigma_R\sigma_T & \sigma_T^2 & 0 \\ 0 & 0 & \sigma_N^2 \end{bmatrix}$$

### 3. Foster/Alfano $P_c$ & The Dilution Region
Diagonalizing $\mathbf{\Sigma}_{2D}$ into principal axes $(\sigma_x, \sigma_y)$ and rotating miss vector $(x_m, y_m)$:
$$P_c \approx \frac{R_c^2}{2 \sigma_x \sigma_y} \exp\left(-\frac{1}{2}\left[\left(\frac{x_m}{\sigma_x}\right)^2 + \left(\frac{y_m}{\sigma_y}\right)^2\right]\right)$$
When uncertainty $\mathbf{\Sigma} \to \infty$, $P_c \to 0$ even for a near direct collision ($d_{\text{miss}} \approx 0$). ECHELON sweeps scale factor $\lambda$ to find peak probability $P_{c,\text{max}}$ and flags the **Dilution Region** whenever $\lambda^* < 0.85$ and $P_{c,\text{max}} \ge 10^{-4}$.

---

## 🚀 Running the Project

### 1. Run the End-to-End Benchmark Pipeline
Propagates all benchmark pairs, runs continuous TCA optimization, predicts learned covariance from the trained ML model, and computes triage metrics dynamically:
```powershell
python run_pipeline.py
```

### 2. Run the Machine Learning Training Pipeline
Regenerate the empirical residual dataset and retrain the HistGBRT quantile regressors:
```powershell
python build_error_dataset.py
python train_quantile_model.py
```

### 3. Run the Automated Test Suite (15/15 Passing)
```powershell
$env:PYTHONPATH = "."
python -m pytest tests
```

### 4. Launch the Interactive Decision Dashboard
```powershell
streamlit run app.py
```
*Dashboard features:*
- **Mode A (Real Propagated Trajectory Screening):** Continuous numerical TCA search on any catalog pair with zero overrides.
- **Mode B (What-If Sensitivity Sandbox):** Stress-test operational triage across hypothetical miss distances and solar storm conditions.
- **Trained Quantile ML Models Tab:** View empirical test-set coverage and feature importances.
- **2D Encounter Plane (B-Plane) & Dilution Curve:** Interactive covariance ellipses and $P_c(\lambda)$ curve.
- **3D Orbit Viewer:** Interactive Plotly 3D Earth sphere with dual-satellite trajectories in TEME.

---

## 📚 Academic & Technical References
1. **Flohrer, T., et al.** (2008). *"Assessment and Categorization of TLE Orbit Errors for the US SSN."* AMOS Conference.
2. **Oltrogge, D. L., et al.** (2014). *"Parametric Characterization of SGP4 Theory and TLE Positional Errors."* AMOS Conference.
3. **NASA Goddard Space Flight Center.** (2016). *"Covariance Manipulation for Conjunction Assessment."* NASA NTRS 20160010497.
4. **NASA NTRS.** (2019). *"Assessment of TLE-based Orbit Determination and Prediction."* NASA NTRS 20190004996.
5. **Embry-Riddle Aeronautical University.** *"Satellite Conjunction Assessment Risk Analysis for Dilution Region."*
