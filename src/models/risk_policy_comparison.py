
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

DATA_DIR = Path("data/processed/temporal_splits")
MODEL_PATH = Path("models/logistic_regression_temporal.joblib")
REPORT_PATH = Path("reports/risk_policy_comparison.json")

CAPACITIES = [0.05, 0.10, 0.15, 0.20]

BASE_FEATURES = [
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
    "type",
]


def prepare_features(df):
    X = df[BASE_FEATURES].copy()
    X["log_amount"] = np.log1p(X["amount"].clip(lower=0))
    X["is_zero_amount"] = (X["amount"] == 0).astype("int8")
    X["is_transfer_or_cashout"] = (
        X["type"].isin(["TRANSFER", "CASH_OUT"])
    ).astype("int8")

    return X[
        [
            "amount",
            "oldbalanceOrg",
            "oldbalanceDest",
            "log_amount",
            "is_zero_amount",
            "is_transfer_or_cashout",
            "type",
        ]
    ]


def choose_threshold(y, scores, capacity):
    # Search thresholds using validation predictions only.
    thresholds = np.unique(scores)
    thresholds = np.append(
        thresholds, np.nextafter(scores.max(), np.inf)
    )

    eligible = []

    for threshold in thresholds:
        pred = scores >= threshold
        alert_rate = float(pred.mean())

        if alert_rate <= capacity:
            tp = int(np.sum(pred & (y == 1)))
            recall = tp / int(np.sum(y)) if np.sum(y) else 0.0

            eligible.append({
                "threshold": float(threshold),
                "validation_alert_rate_pct": alert_rate * 100,
                "validation_recall": float(recall),
            })

    if not eligible:
        raise RuntimeError("No eligible threshold found.")

    # Maximize recall under the capacity constraint.
    # If tied, choose the higher threshold.
    return max(
        eligible,
        key=lambda r: (
            r["validation_recall"],
            r["threshold"],
        ),
    )


def evaluate(y, scores, threshold):
    pred = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(
        y, pred, labels=[0, 1]
    ).ravel()

    return {
        "rows": int(len(y)),
        "alerts": int(pred.sum()),
        "alert_rate_pct": float(pred.mean() * 100),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
        "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "recall": float(tp / (tp + fn)) if tp + fn else None,
    }


def main():
    print("Loading temporal model...")
    model = joblib.load(MODEL_PATH)

    print("Loading validation and test data...")
    validation = pd.read_csv(DATA_DIR / "validation.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")

    results = []

    for transaction_type in ["CASH_OUT", "TRANSFER"]:
        val = validation[validation["type"] == transaction_type]
        test_segment = test[test["type"] == transaction_type]

        print(f"\nScoring {transaction_type}...")

        val_scores = model.predict_proba(
            prepare_features(val)
        )[:, 1]

        test_scores = model.predict_proba(
            prepare_features(test_segment)
        )[:, 1]

        val_y = val["isFraud"].to_numpy(dtype=int)
        test_y = test_segment["isFraud"].to_numpy(dtype=int)

        for capacity in CAPACITIES:
            selection = choose_threshold(
                val_y, val_scores, capacity
            )

            test_result = evaluate(
                test_y,
                test_scores,
                selection["threshold"],
            )

            results.append({
                "type": transaction_type,
                "target_validation_capacity_pct": capacity * 100,
                **selection,
                "test": test_result,
            })

    report = {
        "model": "Chronological Pre-Transaction Logistic Regression",
        "policy": (
            "Maximize validation recall subject to alert capacity; "
            "apply selected threshold unchanged to test data."
        ),
        "results": results,
        "note": (
            "Target capacity applies to validation, not a guarantee "
            "for future alert volume. Test data is evaluation-only."
        ),
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    flat = []
    for item in results:
        row = {
            "type": item["type"],
            "target_capacity_pct": item["target_validation_capacity_pct"],
            "threshold": item["threshold"],
            "validation_recall": item["validation_recall"],
            "validation_alert_rate_pct": item[
                "validation_alert_rate_pct"
            ],
            **{
                f"test_{k}": v
                for k, v in item["test"].items()
            },
        }
        flat.append(row)

    print("\n=== RISK POLICY COMPARISON ===")
    print(pd.DataFrame(flat).round(4).to_string(index=False))
    print(f"\nReport saved: {REPORT_PATH}")


if __name__ == "__main__":
    main()
