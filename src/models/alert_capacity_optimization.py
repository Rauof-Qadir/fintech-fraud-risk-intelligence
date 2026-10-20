
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

DATA_DIR = Path("data/processed/temporal_splits")
MODEL_PATH = Path("models/logistic_regression_temporal.joblib")
REPORT_PATH = Path("reports/alert_capacity_report.json")

ALERT_CAPACITY = 0.10

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


def evaluate(y, scores, threshold):
    pred = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(
        y, pred, labels=[0, 1]
    ).ravel()

    return {
        "threshold": float(threshold),
        "rows": int(len(y)),
        "actual_frauds": int(np.sum(y)),
        "alerts": int(np.sum(pred)),
        "alert_rate_pct": float(np.mean(pred) * 100),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
        "precision": float(
            precision_score(y, pred, zero_division=0)
        ),
        "recall": (
            float(recall_score(y, pred, zero_division=0))
            if np.sum(y) > 0 else None
        ),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "average_precision": (
            float(average_precision_score(y, scores))
            if np.sum(y) > 0 else None
        ),
    }


def select_threshold(y, scores, capacity):
    # Candidate scores are drawn from validation predictions only.
    # Choose the threshold with the highest recall under the alert cap.
    candidates = np.unique(scores)
    candidates = np.append(candidates, np.nextafter(scores.max(), np.inf))

    best = None

    for threshold in candidates:
        pred = scores >= threshold
        alert_rate = float(np.mean(pred))

        if alert_rate > capacity:
            continue

        recall = (
            float(np.sum(pred & (y == 1)) / np.sum(y))
            if np.sum(y) > 0 else 0.0
        )

        candidate = {
            "threshold": float(threshold),
            "validation_alert_rate_pct": alert_rate * 100,
            "validation_recall": recall,
            "validation_alerts": int(np.sum(pred)),
        }

        if (
            best is None
            or candidate["validation_recall"] > best["validation_recall"]
            or (
                candidate["validation_recall"] == best["validation_recall"]
                and candidate["threshold"] > best["threshold"]
            )
        ):
            best = candidate

    if best is None:
        raise RuntimeError("No threshold satisfies the alert capacity.")

    return best


def main():
    print("Loading V4 model...")
    model = joblib.load(MODEL_PATH)

    print("Loading validation data...")
    validation = pd.read_csv(DATA_DIR / "validation.csv")

    print("Loading test data...")
    test = pd.read_csv(DATA_DIR / "test.csv")

    print("Scoring validation data...")
    val_scores = model.predict_proba(
        prepare_features(validation)
    )[:, 1]

    print("Scoring untouched test data...")
    test_scores = model.predict_proba(
        prepare_features(test)
    )[:, 1]

    validation["score"] = val_scores
    test["score"] = test_scores

    results = []
    selected_thresholds = {}

    for transaction_type in ["CASH_OUT", "TRANSFER"]:
        val_segment = validation[
            validation["type"] == transaction_type
        ]
        test_segment = test[
            test["type"] == transaction_type
        ]

        val_y = val_segment["isFraud"].to_numpy(dtype=int)
        val_s = val_segment["score"].to_numpy()

        test_y = test_segment["isFraud"].to_numpy(dtype=int)
        test_s = test_segment["score"].to_numpy()

        selection = select_threshold(
            val_y, val_s, ALERT_CAPACITY
        )
        threshold = selection["threshold"]

        test_result = evaluate(test_y, test_s, threshold)
        test_result["type"] = transaction_type
        test_result["validation_selection"] = selection

        selected_thresholds[transaction_type] = threshold
        results.append(test_result)

    # Compare with the existing single global threshold of 0.19.
    global_results = []

    for transaction_type in ["CASH_OUT", "TRANSFER"]:
        segment = test[test["type"] == transaction_type]
        global_result = evaluate(
            segment["isFraud"].to_numpy(dtype=int),
            segment["score"].to_numpy(),
            0.19,
        )
        global_result["type"] = transaction_type
        global_results.append(global_result)

    report = {
        "alert_capacity_target_per_fraud_related_type_pct": (
            ALERT_CAPACITY * 100
        ),
        "thresholds_selected_on_validation": selected_thresholds,
        "test_results_with_segment_thresholds": results,
        "test_results_with_global_threshold_0_19": global_results,
        "note": (
            "Capacity is an illustrative constraint. Thresholds were "
            "selected using validation data only. Test results are "
            "historical and do not guarantee future performance."
        ),
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n=== SEGMENT THRESHOLDS SELECTED ON VALIDATION ===")
    print(json.dumps(selected_thresholds, indent=2))

    print("\n=== TEST RESULTS: SEGMENT-SPECIFIC THRESHOLDS ===")
    print(pd.DataFrame(results).to_string(index=False))

    print("\n=== TEST RESULTS: GLOBAL THRESHOLD 0.19 ===")
    print(pd.DataFrame(global_results).to_string(index=False))

    print(f"\nReport saved: {REPORT_PATH}")


if __name__ == "__main__":
    main()
