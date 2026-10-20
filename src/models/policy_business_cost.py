
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

DATA_DIR = Path("data/processed/temporal_splits")
MODEL_PATH = Path("models/logistic_regression_temporal.joblib")
POLICY_PATH = Path("reports/risk_policy_comparison.json")
REPORT_PATH = Path("reports/policy_business_cost.json")

FALSE_POSITIVE_COST = 5.0

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
    print("Loading saved policy thresholds...")
    policy = json.loads(
        POLICY_PATH.read_text(encoding="utf-8")
    )

    print("Loading model and chronological test data...")
    model = joblib.load(MODEL_PATH)
    test = pd.read_csv(DATA_DIR / "test.csv")

    test = test[
        test["type"].isin(["CASH_OUT", "TRANSFER"])
    ].copy()

    print("Scoring test transactions...")
    test["score"] = model.predict_proba(
        prepare_features(test)
    )[:, 1]

    results = []

    for capacity in [5.0, 10.0, 15.0, 20.0]:
        segment_results = []

        for transaction_type in ["CASH_OUT", "TRANSFER"]:
            item = next(
                r for r in policy["results"]
                if r["type"] == transaction_type
                and r["target_validation_capacity_pct"] == capacity
            )

            threshold = item["threshold"]

            segment = test[
                test["type"] == transaction_type
            ].copy()

            pred = segment["score"].to_numpy() >= threshold
            y = segment["isFraud"].to_numpy(dtype=int)
            amounts = segment["amount"].to_numpy(dtype=float)

            fp = int(np.sum(pred & (y == 0)))
            fn_mask = (~pred) & (y == 1)
            fn = int(fn_mask.sum())

            false_positive_cost = fp * FALSE_POSITIVE_COST
            missed_fraud_loss = float(amounts[fn_mask].sum())
            total_cost = false_positive_cost + missed_fraud_loss

            segment_results.append({
                "type": transaction_type,
                "threshold": float(threshold),
                "alerts": int(pred.sum()),
                "true_positives": int(np.sum(pred & (y == 1))),
                "false_positives": fp,
                "false_negatives": fn,
                "false_positive_cost": false_positive_cost,
                "missed_fraud_loss": missed_fraud_loss,
                "total_estimated_cost": total_cost,
            })

        results.append({
            "target_validation_capacity_pct": capacity,
            "segments": segment_results,
            "total_alerts": sum(
                r["alerts"] for r in segment_results
            ),
            "total_false_positives": sum(
                r["false_positives"] for r in segment_results
            ),
            "total_false_negatives": sum(
                r["false_negatives"] for r in segment_results
            ),
            "total_fp_cost": sum(
                r["false_positive_cost"] for r in segment_results
            ),
            "total_missed_fraud_loss": sum(
                r["missed_fraud_loss"] for r in segment_results
            ),
            "total_estimated_cost": sum(
                r["total_estimated_cost"] for r in segment_results
            ),
        })

    report = {
        "false_positive_cost_assumption": FALSE_POSITIVE_COST,
        "missed_fraud_loss_assumption": (
            "Full transaction amount for each missed fraud"
        ),
        "results": results,
        "warning": (
            "Illustrative costs only, not verified bank economics. "
            "Thresholds came from validation; test data is evaluation-only."
        ),
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    table = pd.DataFrame([
        {
            "capacity_pct": r["target_validation_capacity_pct"],
            "alerts": r["total_alerts"],
            "false_positives": r["total_false_positives"],
            "missed_frauds": r["total_false_negatives"],
            "fp_cost_usd": r["total_fp_cost"],
            "missed_fraud_loss_usd": r["total_missed_fraud_loss"],
            "total_estimated_cost_usd": r["total_estimated_cost"],
        }
        for r in results
    ])

    print("\n=== TEST-PERIOD BUSINESS COST COMPARISON ===")
    print(table.round(2).to_string(index=False))
    print(f"\nReport saved: {REPORT_PATH}")


if __name__ == "__main__":
    main()
