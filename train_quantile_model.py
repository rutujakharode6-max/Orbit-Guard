"""
Quantile Regression Training & Validation Pipeline
Trains Gradient Boosted Quantile Regressors (q=0.50, q=0.84, q=0.95) for Radial, In-Track, and Cross-Track SGP4 errors.
Evaluates empirical coverage and calibration against held-out test data.
Saves model artifacts to data/trained_quantile_models.joblib.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

# Suppress Windows joblib cpu count warning
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
INPUT_CSV = os.path.join(DATA_DIR, "training_residuals.csv")
MODEL_ARTIFACT = os.path.join(DATA_DIR, "trained_quantile_models.joblib")
METRICS_JSON = os.path.join(DATA_DIR, "model_metrics.json")

FEATURE_COLS = [
    "delta_t_hours",
    "altitude_km",
    "perigee_altitude_km",
    "eccentricity",
    "inclination_deg",
    "bstar",
    "f107",
    "kp",
    "is_maneuvering"
]

QUANTILES = {
    "median": 0.50,
    "1_sigma": 0.8413,   # 84.13th percentile represents the upper 1-sigma bound of positive magnitude
    "2_sigma": 0.9545,   # 95.45th percentile represents 2-sigma bound
    "3_sigma": 0.9973    # 99.73th percentile
}

def train_and_evaluate():
    print("=" * 80)
    print(" TRAINING DATA-DRIVEN QUANTILE REGRESSION ERROR MODELS (HIST-GBRT)")
    print("=" * 80)

    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Training data not found at {INPUT_CSV}. Run build_error_dataset.py first.")

    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} samples from {INPUT_CSV}")

    X = df[FEATURE_COLS]
    targets = {
        "radial": df["err_radial_km"],
        "intrack": df["err_intrack_km"],
        "crosstrack": df["err_crosstrack_km"]
    }

    # Split train/test (80/20) with fixed random seed
    train_idx, test_idx = train_test_split(df.index, test_size=0.20, random_state=42)
    X_train = X.loc[train_idx]
    X_test = X.loc[test_idx]

    trained_models = {}
    metrics_report = {
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "features": FEATURE_COLS,
        "axes": {}
    }

    for axis_name, y in targets.items():
        print(f"\n[-] Training Quantile Models for [{axis_name.upper()}] Error...")
        y_train = y.loc[train_idx]
        y_test = y.loc[test_idx]

        trained_models[axis_name] = {}
        axis_metrics = {}

        for q_label, q_val in QUANTILES.items():
            model = HistGradientBoostingRegressor(
                loss="quantile",
                quantile=q_val,
                max_iter=100,
                max_leaf_nodes=31,
                min_samples_leaf=20,
                learning_rate=0.08,
                random_state=42
            )
            model.fit(X_train, y_train)
            trained_models[axis_name][q_label] = model

            # Evaluate on held-out test set
            y_pred = model.predict(X_test)
            
            # Coverage: fraction of held-out real errors <= predicted bound
            empirical_coverage = float(np.mean(y_test <= y_pred) * 100.0)
            mae = float(mean_absolute_error(y_test, y_pred))
            rmse = float(root_mean_squared_error(y_test, y_pred))

            axis_metrics[q_label] = {
                "nominal_target_pct": round(q_val * 100.0, 2),
                "empirical_coverage_pct": round(empirical_coverage, 2),
                "mae_km": round(mae, 4),
                "rmse_km": round(rmse, 4),
                "mean_pred_km": round(float(np.mean(y_pred)), 4)
            }

            print(f"    * {q_label.ljust(8)} (q={q_val:.4f}): Empirical Coverage = {empirical_coverage:.1f}% (Target: {q_val*100:.1f}%), MAE = {mae*1000:.1f} m")

        # Uncalibrated naive baseline comparison: fixed static standard deviation
        naive_sigma = float(np.mean(y_train) * 1.5)
        naive_coverage = float(np.mean(y_test <= naive_sigma) * 100.0)
        axis_metrics["naive_baseline"] = {
            "fixed_sigma_km": round(naive_sigma, 4),
            "empirical_coverage_pct": round(naive_coverage, 2)
        }

        metrics_report["axes"][axis_name] = axis_metrics

    # Save artifact
    bundle = {
        "models": trained_models,
        "features": FEATURE_COLS,
        "quantiles": QUANTILES,
        "metrics": metrics_report
    }
    joblib.dump(bundle, MODEL_ARTIFACT)
    print(f"\n[+] Successfully saved trained quantile models to {MODEL_ARTIFACT}")

    with open(METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(metrics_report, f, indent=2)
    print(f"[+] Saved calibration metrics to {METRICS_JSON}")

    print("\n" + "=" * 80)
    print(" QUANTILE COVERAGE SUMMARY (TEST SET VALIDATION)")
    print("=" * 80)
    for axis, data in metrics_report["axes"].items():
        print(f"Axis: {axis.upper():12} | 1-Sigma Bound Coverage: {data['1_sigma']['empirical_coverage_pct']}% | 2-Sigma Coverage: {data['2_sigma']['empirical_coverage_pct']}%")
    print("=" * 80 + "\n")

    return bundle

if __name__ == "__main__":
    train_and_evaluate()
