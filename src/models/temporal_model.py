
import json
from pathlib import Path

import joblib
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_DIR = Path("data/processed/temporal_splits")
MODEL_PATH = Path("models/logistic_regression_temporal.joblib")
REPORT_PATH = Path("reports/temporal_model_report.json")

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
    result = df[[
        "amount",
        "oldbalanceOrg",
        "oldbalanceDest",
        "type",
    ]].copy()

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


def evaluate_cost(y_true, probabilities, amounts, threshold):
    predictions = probabilities >= threshold

    false_positives = (predictions & (y_true == 0))
    false_negatives = (~predictions & (y_true == 1))

    fp_count = int(false_positives.sum())
    fn_count = int(false_negatives.sum())

    fp_cost = fp_count * FP_COST
    missed_loss = float(amounts[false_negatives].sum())

    return {
        "threshold": float(threshold),
        "alerts": int(predictions.sum()),
        "false_positives": fp_count,
        "false_negatives": fn_count,
        "false_positive_cost_usd": float(fp_cost),
        "missed_fraud_loss_usd": missed_loss,
        "total_estimated_cost_usd": float(fp_cost + missed_loss),
    }


def main():
    print("Loading chronological datasets...")

    train = pd.read_csv(DATA_DIR / "train.csv")
    validation = pd.read_csv(DATA_DIR / "validation.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")

    y_train = train["isFraud"].astype(int)
    y_val = validation["isFraud"].astype(int).to_numpy()
    y_test = test["isFraud"].astype(int).to_numpy()

    print(f"Training rows:   {len(train):,}")
    print(f"Validation rows: {len(validation):,}")
    print(f"Test rows:       {len(test):,}")

    print("\nPreparing pre-transaction features...")
    X_train = prepare_features(train)
    X_val = prepare_features(validation)
    X_test = prepare_features(test)

    del train

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

    print("\nTraining on past transactions...")
    model.fit(X_train, y_train)
    del X_train, y_train

    print("Scoring validation transactions...")
    val_probabilities = model.predict_proba(X_val)[:, 1]
    val_amounts = validation["amount"].to_numpy()

    print("Selecting threshold using validation only...")
    best_result = None

    for threshold in np.linspace(0.01, 0.99, 99):
        result = evaluate_cost(
            y_val,
            val_probabilities,
            val_amounts,
            threshold,
        )

        if (
            best_result is None
            or result["total_estimated_cost_usd"]
            < best_result["total_estimated_cost_usd"]
        ):
            best_result = result

    selected_threshold = best_result["threshold"]
    print(f"Selected threshold: {selected_threshold:.4f}")

    print("\nEvaluating the later, untouched test period...")
    test_probabilities = model.predict_proba(X_test)[:, 1]
    test_amounts = test["amount"].to_numpy()

    optimized_cost = evaluate_cost(
        y_test,
        test_probabilities,
        test_amounts,
        selected_threshold,
    )

    predictions = (
        test_probabilities >= selected_threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    ).ravel()

    report = {
        "model": "Chronological Pre-Transaction Logistic Regression",
        "split_strategy": {
            "training_steps": "1-520",
            "validation_steps": "521-631",
            "test_steps": "632-743",
        },
        "features": FEATURES,
        "threshold_selection_on_validation": best_result,
        "test_metrics": {
            **optimized_cost,
            "rows": int(len(test)),
            "actual_fraud": int(y_test.sum()),
            "true_positives": int(tp),
            "true_negatives": int(tn),
            "precision": float(
                precision_score(y_test, predictions, zero_division=0)
            ),
            "recall": float(
                recall_score(y_test, predictions, zero_division=0)
            ),
            "f1": float(
                f1_score(y_test, predictions, zero_division=0)
            ),
            "average_precision": float(
                average_precision_score(y_test, test_probabilities)
            ),
        },
        "business_assumptions": {
            "false_positive_cost_usd": FP_COST,
            "missed_fraud_loss": "Full transaction amount",
            "note": (
                "Illustrative assumptions, not verified bank costs. "
                "Evaluation is historical and does not guarantee "
                "future production performance."
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

    print("\n=== TEMPORAL MODEL REPORT ===")
    print(json.dumps(report, indent=2))
    print(f"\nModel saved: {MODEL_PATH}")
    print(f"Report saved: {REPORT_PATH}")


if __name__ == "__main__":
    main()
