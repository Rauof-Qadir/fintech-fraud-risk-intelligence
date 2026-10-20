
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib


TRAIN_PATH = Path("data/processed/splits/train.csv")
TEST_PATH = Path("data/processed/splits/test.csv")
MODEL_PATH = Path("models/logistic_regression_pre_transaction.joblib")
REPORT_PATH = Path("reports/pre_transaction_report.json")

FP_COST = 5.0

NUMERIC_FEATURES = [
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
    "log_amount",
    "is_zero_amount",
    "is_transfer_or_cashout",
]

CATEGORICAL_FEATURES = ["type"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def prepare_features(df):
    result = df.copy()

    result["log_amount"] = np.log1p(
        result["amount"].clip(lower=0)
    )
    result["is_zero_amount"] = (
        result["amount"] == 0
    ).astype("int8")
    result["is_transfer_or_cashout"] = (
        result["type"].isin(["TRANSFER", "CASH_OUT"])
    ).astype("int8")

    return result[FEATURES]


def business_cost(y_true, probabilities, amounts, threshold):
    predictions = probabilities >= threshold

    false_positives = (
        (predictions == 1) & (y_true == 0)
    )
    false_negatives = (
        (predictions == 0) & (y_true == 1)
    )

    fp_cost = int(false_positives.sum()) * FP_COST
    missed_fraud_loss = float(amounts[false_negatives].sum())

    return {
        "threshold": float(threshold),
        "alerts": int(predictions.sum()),
        "false_positives": int(false_positives.sum()),
        "false_negatives": int(false_negatives.sum()),
        "false_positive_cost_usd": float(fp_cost),
        "missed_fraud_loss_usd": missed_fraud_loss,
        "total_estimated_cost_usd": (
            float(fp_cost) + missed_fraud_loss
        ),
    }


def main():
    print("Loading train and test data...")
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    y = train["isFraud"].astype(int)
    y_test = test["isFraud"].astype(int)

    X = prepare_features(train)
    X_test = prepare_features(test)

    # Test set is kept untouched during threshold selection.
    X_fit, X_val, y_fit, y_val, amount_fit, amount_val = (
        train_test_split(
            X,
            y,
            train["amount"].to_numpy(),
            test_size=0.20,
            random_state=42,
            stratify=y,
        )
    )

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessing = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])

    model = Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        )),
    ])

    print("Training validation model...")
    model.fit(X_fit, y_fit)
    val_probs = model.predict_proba(X_val)[:, 1]

    print("Selecting threshold using validation data...")
    best_result = None

    for threshold in np.linspace(0.01, 0.99, 99):
        result = business_cost(
            y_val.to_numpy(),
            val_probs,
            amount_val,
            threshold,
        )

        if (
            best_result is None
            or result["total_estimated_cost_usd"]
            < best_result["total_estimated_cost_usd"]
        ):
            best_result = result

    threshold = best_result["threshold"]
    print(f"Selected threshold: {threshold:.4f}")

    print("Refitting model on all training data...")
    model.fit(X, y)

    print("Evaluating untouched test data...")
    test_probs = model.predict_proba(X_test)[:, 1]
    test_amounts = test["amount"].to_numpy()

    optimized = business_cost(
        y_test.to_numpy(),
        test_probs,
        test_amounts,
        threshold,
    )

    predictions = (test_probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    ).ravel()

    report = {
        "model": "Pre-Transaction Logistic Regression",
        "features": FEATURES,
        "excluded_post_transaction_features": [
            "newbalanceOrig",
            "newbalanceDest",
            "sender_balance_change",
            "receiver_balance_change",
            "sender_error",
            "receiver_error",
        ],
        "business_assumptions": {
            "false_positive_cost_usd": FP_COST,
            "missed_fraud_loss": "Full transaction amount",
            "note": "Illustrative assumptions, not verified bank costs.",
        },
        "validation_threshold_selection": best_result,
        "test_results": {
            **optimized,
            "true_positives": int(tp),
            "true_negatives": int(tn),
            "precision": float(
                precision_score(y_test, predictions, zero_division=0)
            ),
            "recall": float(
                recall_score(y_test, predictions, zero_division=0)
            ),
            "f1": float(f1_score(y_test, predictions, zero_division=0)),
            "average_precision": float(
                average_precision_score(y_test, test_probs)
            ),
        },
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2))
    print(f"\nModel saved: {MODEL_PATH}")
    print(f"Report saved: {REPORT_PATH}")


if __name__ == "__main__":
    main()
