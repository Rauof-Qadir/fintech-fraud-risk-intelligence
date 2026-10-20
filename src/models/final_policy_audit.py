
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

DATA_DIR = Path("data/processed/temporal_splits")
MODEL_PATH = Path("models/logistic_regression_temporal.joblib")
POLICY_PATH = Path("reports/risk_policy_comparison.json")
REPORT_PATH = Path("reports/final_policy_audit.json")

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


def evaluate_segment(segment, threshold):
    y = segment["isFraud"].to_numpy(dtype=int)
    scores = segment["score"].to_numpy()
    pred = scores >= threshold

    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum((~pred) & (y == 1)))
    tn = int(np.sum((~pred) & (y == 0)))

    fraud_amounts = segment["amount"].to_numpy(dtype=float)
    missed_loss = float(fraud_amounts[(~pred) & (y == 1)].sum())

    return {
        "rows": int(len(segment)),
        "actual_frauds": int(y.sum()),
        "threshold": float(threshold),
        "alerts": int(pred.sum()),
        "alert_rate_pct": float(pred.mean() * 100),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "precision_pct": float(tp / (tp + fp) * 100)
        if tp + fp else 0.0,
        "recall_pct": float(tp / (tp + fn) * 100)
        if tp + fn else None,
        "average_precision": float(
            average_precision_score(y, scores)
        ) if y.sum() else None,
        "illustrative_cost_fp_5_usd": float(fp * 5),
        "missed_fraud_amount_usd": missed_loss,
        "illustrative_total_cost_usd": float(fp * 5 + missed_loss),
    }


def main():
    print("Loading model, policy, and untouched test data...")
    model = joblib.load(MODEL_PATH)
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    test = pd.read_csv(DATA_DIR / "test.csv")

    test = test[
        test["type"].isin(["CASH_OUT", "TRANSFER"])
    ].copy()

    print("Scoring test data...")
    test["score"] = model.predict_proba(
        prepare_features(test)
    )[:, 1]

    results = []

    for transaction_type in ["CASH_OUT", "TRANSFER"]:
        segment = test[test["type"] == transaction_type].copy()

        policy_item = next(
            item for item in policy["results"]
            if item["type"] == transaction_type
            and item["target_validation_capacity_pct"] == 20.0
        )

        selected_threshold = policy_item["threshold"]

        # Recommended segment-specific threshold.
        selected_result = evaluate_segment(
            segment, selected_threshold
        )
        selected_result["policy"] = "segment_specific_20pct_target"
        selected_result["type"] = transaction_type
        selected_result["validation_alert_rate_pct"] = (
            policy_item["validation_alert_rate_pct"]
        )
        results.append(selected_result)

        # Existing global threshold baseline.
        global_result = evaluate_segment(segment, 0.19)
        global_result["policy"] = "global_threshold_0.19"
        global_result["type"] = transaction_type
        global_result["validation_alert_rate_pct"] = None
        results.append(global_result)

    df_results = pd.DataFrame(results)

    summary = (
        df_results.groupby("policy", as_index=False)
        .agg(
            total_alerts=("alerts", "sum"),
            total_false_positives=("false_positives", "sum"),
            total_false_negatives=("false_negatives", "sum"),
            total_fp_cost_usd=("illustrative_cost_fp_5_usd", "sum"),
            total_missed_fraud_amount_usd=("missed_fraud_amount_usd", "sum"),
            total_estimated_cost_usd=("illustrative_total_cost_usd", "sum"),
        )
    )

    report = {
        "test_period_rows": int(len(test)),
        "recommended_policy": (
            "Segment-specific thresholds selected on validation "
            "for a 20% target alert rate per segment."
        ),
        "global_baseline_threshold": 0.19,
        "segment_results": results,
        "policy_summary": summary.to_dict(orient="records"),
        "cost_assumptions": {
            "false_positive_cost_usd": 5,
            "missed_fraud_loss": "Full transaction amount",
            "warning": "Illustrative costs, not verified bank economics.",
        },
        "limitations": [
            "Historical test performance does not guarantee future results.",
            "A validation alert-rate target does not guarantee test capacity.",
            "Thresholds should be recalibrated only using an approved validation process.",
            "The test set must not be used to tune thresholds.",
        ],
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n=== SEGMENT-LEVEL POLICY AUDIT ===")
    print(
        df_results[
            [
                "type",
                "policy",
                "threshold",
                "alerts",
                "alert_rate_pct",
                "true_positives",
                "false_positives",
                "false_negatives",
                "precision_pct",
                "recall_pct",
                "illustrative_total_cost_usd",
            ]
        ].round(4).to_string(index=False)
    )

    print("\n=== TOTAL POLICY COMPARISON ===")
    print(summary.round(2).to_string(index=False))
    print(f"\nReport saved: {REPORT_PATH}")


if __name__ == "__main__":
    main()
