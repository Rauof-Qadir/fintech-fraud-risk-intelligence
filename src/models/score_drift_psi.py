
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

DATA_DIR = Path("data/processed/temporal_splits")
MODEL_PATH = Path("models/logistic_regression_temporal.joblib")
POLICY_PATH = Path("reports/risk_policy_comparison.json")
REPORT_PATH = Path("reports/score_drift_psi_report.json")

FEATURES = [
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
    "log_amount",
    "is_zero_amount",
    "is_transfer_or_cashout",
    "type",
]

EPSILON = 1e-6


def prepare_features(df):
    X = df[
        ["amount", "oldbalanceOrg", "oldbalanceDest", "type"]
    ].copy()

    X["log_amount"] = np.log1p(X["amount"].clip(lower=0))
    X["is_zero_amount"] = (X["amount"] == 0).astype("int8")
    X["is_transfer_or_cashout"] = (
        X["type"].isin(["TRANSFER", "CASH_OUT"])
    ).astype("int8")

    return X[FEATURES]


def psi_from_scores(reference_scores, current_scores, bins=10):
    # Create fixed score bins using reference data only.
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(
        np.quantile(reference_scores, quantiles)
    )

    # Handle tied or constant scores safely.
    if len(edges) < 3:
        edges = np.linspace(0, 1, bins + 1)

    edges[0] = min(edges[0], 0.0)
    edges[-1] = max(edges[-1], 1.0)

    ref_counts, _ = np.histogram(reference_scores, bins=edges)
    cur_counts, _ = np.histogram(current_scores, bins=edges)

    ref_pct = ref_counts / max(ref_counts.sum(), 1)
    cur_pct = cur_counts / max(cur_counts.sum(), 1)

    ref_pct = np.clip(ref_pct, EPSILON, None)
    cur_pct = np.clip(cur_pct, EPSILON, None)

    psi = np.sum(
        (cur_pct - ref_pct) * np.log(cur_pct / ref_pct)
    )

    return float(psi)


def psi_status(value):
    if value < 0.10:
        return "LOW"
    if value < 0.25:
        return "MODERATE"
    return "HIGH"


def main():
    print("Loading model and validation-selected thresholds...")
    model = joblib.load(MODEL_PATH)
    policy = json.loads(
        POLICY_PATH.read_text(encoding="utf-8")
    )

    thresholds = {
        (r["type"], r["target_validation_capacity_pct"]):
            r["threshold"]
        for r in policy["results"]
    }

    # Validation is the reference distribution.
    # Train scores are in-sample, so we avoid using them as the
    # reference for a fair score-distribution comparison.
    validation = pd.read_csv(DATA_DIR / "validation.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")

    print("Scoring validation reference and test periods...")
    validation["score"] = model.predict_proba(
        prepare_features(validation)
    )[:, 1]

    test["score"] = model.predict_proba(
        prepare_features(test)
    )[:, 1]

    results = []

    for transaction_type in ["CASH_OUT", "TRANSFER"]:
        ref = validation[
            validation["type"] == transaction_type
        ].copy()

        cur = test[
            test["type"] == transaction_type
        ].copy()

        psi = psi_from_scores(
            ref["score"].to_numpy(),
            cur["score"].to_numpy(),
        )

        threshold = thresholds[(transaction_type, 20.0)]

        results.append({
            "type": transaction_type,
            "reference_period": "validation",
            "current_period": "test",
            "reference_rows": int(len(ref)),
            "current_rows": int(len(cur)),
            "reference_mean_score": float(ref["score"].mean()),
            "current_mean_score": float(cur["score"].mean()),
            "reference_alert_rate_pct": float(
                (ref["score"] >= threshold).mean() * 100
            ),
            "current_alert_rate_pct": float(
                (cur["score"] >= threshold).mean() * 100
            ),
            "psi": psi,
            "psi_status": psi_status(psi),
            "threshold": float(threshold),
            "reference_fraud_rate_pct": float(
                ref["isFraud"].mean() * 100
            ),
            "current_fraud_rate_pct": float(
                cur["isFraud"].mean() * 100
            ),
        })

    report = {
        "method": (
            "Score-distribution PSI using validation as reference "
            "and test as current period."
        ),
        "threshold_guidelines": {
            "low": "PSI < 0.10",
            "moderate": "0.10 <= PSI < 0.25",
            "high": "PSI >= 0.25",
        },
        "results": results,
        "limitations": [
            "PSI measures score-distribution change, not model accuracy.",
            "Fraud prevalence changes can affect score distributions.",
            "Historical test results do not guarantee future performance.",
            "Thresholds were selected on validation data.",
        ],
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n=== VALIDATION VS TEST SCORE DRIFT ===")
    print(pd.DataFrame(results).round(4).to_string(index=False))
    print(f"\nReport saved: {REPORT_PATH}")


if __name__ == "__main__":
    main()
