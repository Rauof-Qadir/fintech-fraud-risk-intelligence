
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

DATA_DIR = Path("data/processed/temporal_splits")
MODEL_PATH = Path("models/logistic_regression_temporal.joblib")
POLICY_PATH = Path("reports/risk_policy_comparison.json")
REPORT_PATH = Path("reports/alert_monitoring_report.json")

FEATURES = [
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
    "log_amount",
    "is_zero_amount",
    "is_transfer_or_cashout",
    "type",
]


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


def main():
    print("Loading model and selected policy thresholds...")
    model = joblib.load(MODEL_PATH)
    policy = json.loads(
        POLICY_PATH.read_text(encoding="utf-8")
    )

    thresholds = {
        (r["type"], r["target_validation_capacity_pct"]):
            r["threshold"]
        for r in policy["results"]
    }

    summaries = []
    by_step = []

    for split_name in ["train", "validation", "test"]:
        path = DATA_DIR / f"{split_name}.csv"
        print(f"\nMonitoring {split_name}: {path}")

        df = pd.read_csv(path)

        scores = model.predict_proba(
            prepare_features(df)
        )[:, 1]

        df["risk_score"] = scores

        for transaction_type in ["CASH_OUT", "TRANSFER"]:
            segment = df[df["type"] == transaction_type].copy()

            if segment.empty:
                continue

            # Use the 20% validation-target policy for monitoring.
            threshold = thresholds[(transaction_type, 20.0)]
            alerts = segment["risk_score"] >= threshold

            summaries.append({
                "split": split_name,
                "type": transaction_type,
                "rows": int(len(segment)),
                "mean_score": float(segment["risk_score"].mean()),
                "median_score": float(segment["risk_score"].median()),
                "p95_score": float(segment["risk_score"].quantile(0.95)),
                "threshold": float(threshold),
                "alerts": int(alerts.sum()),
                "alert_rate_pct": float(alerts.mean() * 100),
                # Available for historical evaluation, not live monitoring.
                "fraud_rate_pct": float(
                    segment["isFraud"].mean() * 100
                ),
            })

            if split_name == "test":
                segment["alerts"] = alerts.astype(int)

                step_summary = (
                    segment.groupby("step")
                    .agg(
                        transactions=("step", "size"),
                        mean_risk_score=("risk_score", "mean"),
                        alerts=("alerts", "sum"),
                        actual_frauds=("isFraud", "sum"),
                    )
                    .reset_index()
                )

                step_summary["type"] = transaction_type
                step_summary["alert_rate_pct"] = (
                    step_summary["alerts"]
                    / step_summary["transactions"] * 100
                )
                step_summary["fraud_rate_pct"] = (
                    step_summary["actual_frauds"]
                    / step_summary["transactions"] * 100
                )

                by_step.extend(
                    step_summary.to_dict(orient="records")
                )

    report = {
        "policy": "20% target capacity thresholds selected on validation",
        "split_summary": summaries,
        "test_by_step": by_step,
        "limitations": [
            "Historical fraud labels may not be immediately available in production.",
            "The selected validation thresholds do not guarantee a fixed future alert rate.",
            "Score and alert shifts are monitoring signals, not proof of model failure.",
        ],
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n=== MONITORING SUMMARY ===")
    print(pd.DataFrame(summaries).round(4).to_string(index=False))

    print(f"\nReport saved: {REPORT_PATH}")


if __name__ == "__main__":
    main()
