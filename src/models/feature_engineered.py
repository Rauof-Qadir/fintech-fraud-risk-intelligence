
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.features import engineer_features


ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = ROOT / "data/processed/splits/train.csv"
TEST_PATH = ROOT / "data/processed/splits/test.csv"
MODEL_PATH = ROOT / "models/logistic_regression_feature_engineered.joblib"
REPORT_PATH = ROOT / "reports/feature_engineered_cost_report.json"

TARGET = "isFraud"
NUMERIC_FEATURES = [
    "amount",
    "sender_balance_change",
    "receiver_balance_change",
    "sender_error",
    "receiver_error",
    "log_amount",
    "is_zero_amount",
    "is_transfer_or_cashout",
]
CATEGORICAL_FEATURES = ["type"]

FP_COST = 5.0


def build_model():
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

    return Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        )),
    ])


def prepare_features(df):
    result = engineer_features(df)
    return result[NUMERIC_FEATURES + CATEGORICAL_FEATURES]


def select_cost_threshold(y, amounts, scores):
    """Choose a threshold using validation data only."""
    y = np.asarray(y, dtype=np.int8)
    amounts = np.asarray(amounts, dtype=float)
    scores = np.asarray(scores, dtype=float)

    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    sorted_y = y[order]
    sorted_amounts = amounts[order]

    # Prefix totals represent transactions below each candidate threshold.
    fraud_losses = sorted_y * sorted_amounts
    prefix_fraud_loss = np.concatenate(
        ([0.0], np.cumsum(fraud_losses))
    )
    prefix_fraud_count = np.concatenate(
        ([0], np.cumsum(sorted_y))
    )

    thresholds = np.unique(np.concatenate((
        [0.0],
        scores,
        [np.nextafter(float(scores.max()), np.inf)],
    )))

    positions = np.searchsorted(
        sorted_scores, thresholds, side="left"
    )

    total_rows = len(y)
    total_fraud = int(y.sum())
    alerts = total_rows - positions
    true_positives = total_fraud - prefix_fraud_count[positions]
    false_positives = alerts - true_positives
    false_negatives = prefix_fraud_count[positions]

    missed_fraud_loss = prefix_fraud_loss[positions]
    fp_cost = false_positives * FP_COST
    total_cost = fp_cost + missed_fraud_loss

    # Lowest cost wins; ties prefer fewer alerts.
    best_index = np.lexsort((alerts, total_cost))[0]

    return {
        "threshold": float(thresholds[best_index]),
        "validation_total_cost_usd": float(total_cost[best_index]),
        "validation_false_positive_cost_usd": float(fp_cost[best_index]),
        "validation_missed_fraud_loss_usd": float(
            missed_fraud_loss[best_index]
        ),
        "validation_alerts": int(alerts[best_index]),
        "validation_true_positives": int(true_positives[best_index]),
        "validation_false_positives": int(false_positives[best_index]),
        "validation_false_negatives": int(false_negatives[best_index]),
    }


def evaluate(y, amounts, scores, threshold):
    y = np.asarray(y, dtype=np.int8)
    amounts = np.asarray(amounts, dtype=float)
    predictions = (np.asarray(scores) >= threshold).astype(np.int8)

    tp = int(((y == 1) & (predictions == 1)).sum())
    fp = int(((y == 0) & (predictions == 1)).sum())
    tn = int(((y == 0) & (predictions == 0)).sum())
    fn = int(((y == 1) & (predictions == 0)).sum())

    fp_cost = fp * FP_COST
    missed_loss = float(amounts[(y == 1) & (predictions == 0)].sum())

    return {
        "threshold": float(threshold),
        "rows": int(len(y)),
        "actual_fraud": int(y.sum()),
        "alerts": int(predictions.sum()),
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "average_precision": float(average_precision_score(y, scores)),
        "false_positive_cost_usd": float(fp_cost),
        "missed_fraud_loss_usd": missed_loss,
        "total_estimated_cost_usd": float(fp_cost + missed_loss),
    }


def main():
    print("Loading training data...")
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    # Keep test data out of threshold selection.
    fit_df, val_df = train_test_split(
        train,
        test_size=0.20,
        random_state=42,
        stratify=train[TARGET],
    )

    print("Training validation model...")
    validation_model = build_model()
    validation_model.fit(
        prepare_features(fit_df),
        fit_df[TARGET],
    )

    validation_scores = validation_model.predict_proba(
        prepare_features(val_df)
    )[:, 1]

    selected = select_cost_threshold(
        val_df[TARGET],
        val_df["amount"],
        validation_scores,
    )
    threshold = selected["threshold"]

    print(f"Selected validation threshold: {threshold:.8f}")
    print("Refitting final model on all training data...")

    final_model = build_model()
    final_model.fit(prepare_features(train), train[TARGET])

    print("Evaluating untouched test data...")
    test_scores = final_model.predict_proba(
        prepare_features(test)
    )[:, 1]

    optimized = evaluate(
        test[TARGET], test["amount"], test_scores, threshold
    )
    default = evaluate(
        test[TARGET], test["amount"], test_scores, 0.5
    )

    report = {
        "model": "Feature-engineered Logistic Regression",
        "features": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
        "business_assumptions": {
            "false_positive_investigation_cost_usd": FP_COST,
            "missed_fraud_loss": "Full transaction amount",
            "note": (
                "Illustrative assumptions, not verified bank costs. "
                "Estimated cost is not the same as confirmed realized loss."
            ),
        },
        "threshold_selection": selected,
        "untouched_test_evaluation": {
            "optimized_threshold": optimized,
            "threshold_0_5": default,
            "estimated_savings_vs_threshold_0_5_usd": float(
                default["total_estimated_cost_usd"]
                - optimized["total_estimated_cost_usd"]
            ),
        },
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(final_model, MODEL_PATH)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2))
    print(f"\nModel saved: {MODEL_PATH}")
    print(f"Report saved: {REPORT_PATH}")


if __name__ == "__main__":
    main()
