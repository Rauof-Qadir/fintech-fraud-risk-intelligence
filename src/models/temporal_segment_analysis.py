
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

DATA_PATH = Path("data/processed/temporal_splits/test.csv")
MODEL_PATH = Path("models/logistic_regression_temporal.joblib")
REPORT_PATH = Path("reports/temporal_segment_report.json")

THRESHOLD = 0.19


def prepare_features(df):
    result = df[
        ["amount", "oldbalanceOrg", "oldbalanceDest", "type"]
    ].copy()

    result["log_amount"] = np.log1p(
        result["amount"].clip(lower=0)
    )
    result["is_zero_amount"] = (
        result["amount"] == 0
    ).astype("int8")
    result["is_transfer_or_cashout"] = (
        result["type"].isin(["TRANSFER", "CASH_OUT"])
    ).astype("int8")

    return result[
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


def main():
    print("Loading model and chronological test data...")
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)

    y = df["isFraud"].astype(int).to_numpy()
    probabilities = model.predict_proba(
        prepare_features(df)
    )[:, 1]

    df["fraud_probability"] = probabilities
    df["prediction"] = (
        df["fraud_probability"] >= THRESHOLD
    ).astype(int)

    overall = []
    by_type = []

    for segment_name, segment in [
        ("ALL", df),
        *list(df.groupby("type", sort=True)),
    ]:
        actual = segment["isFraud"].astype(int).to_numpy()
        predicted = segment["prediction"].to_numpy()
        scores = segment["fraud_probability"].to_numpy()

        tn, fp, fn, tp = confusion_matrix(
            actual, predicted, labels=[0, 1]
        ).ravel()

        fraud_count = int(actual.sum())
        alert_count = int(predicted.sum())

        result = {
            "segment": str(segment_name),
            "rows": int(len(segment)),
            "actual_frauds": fraud_count,
            "fraud_rate_pct": round(
                float(actual.mean() * 100), 5
            ),
            "alerts": alert_count,
            "alert_rate_pct": round(
                float(predicted.mean() * 100), 4
            ),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_negatives": int(tn),
            "precision": float(
                precision_score(actual, predicted, zero_division=0)
            ),
            "recall": (
                float(recall_score(actual, predicted, zero_division=0))
                if fraud_count else None
            ),
            "f1": float(
                f1_score(actual, predicted, zero_division=0)
            ),
            "average_precision": (
                float(average_precision_score(actual, scores))
                if fraud_count else None
            ),
            "median_fraud_score": (
                float(np.median(scores[actual == 1]))
                if fraud_count else None
            ),
            "median_legitimate_score": (
                float(np.median(scores[actual == 0]))
                if (actual == 0).any() else None
            ),
        }

        if segment_name == "ALL":
            overall.append(result)
        else:
            by_type.append(result)

    report = {
        "threshold": THRESHOLD,
        "overall": overall[0],
        "by_transaction_type": by_type,
        "note": (
            "Threshold was selected on validation data. "
            "Test data is used only for this audit. "
            "Results apply to this historical dataset."
        ),
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n=== OVERALL TEST PERFORMANCE ===")
    print(pd.DataFrame(overall).round(5).to_string(index=False))

    print("\n=== PERFORMANCE BY TRANSACTION TYPE ===")
    print(pd.DataFrame(by_type).round(5).to_string(index=False))

    print(f"\nReport saved: {REPORT_PATH}")


if __name__ == "__main__":
    main()
