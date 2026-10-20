
from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


FEATURES = ["amount", "type"]
TARGET = "isFraud"


def train_baseline(
    train_path="data/processed/splits/train.csv",
    test_path="data/processed/splits/test.csv",
):
    train = pd.read_csv(train_path, usecols=FEATURES + [TARGET])
    test = pd.read_csv(test_path, usecols=FEATURES + [TARGET])

    X_train = train[FEATURES]
    y_train = train[TARGET]
    X_test = test[FEATURES]
    y_test = test[TARGET]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessing = ColumnTransformer([
        ("numeric", numeric_pipeline, ["amount"]),
        ("categorical", categorical_pipeline, ["type"]),
    ])

    model = Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        )),
    ])

    model.fit(X_train, y_train)

    # Default decision threshold: 0.50.
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    tn, fp, fn, tp = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    ).ravel()

    report = {
        "model": "Logistic Regression",
        "features": FEATURES,
        "threshold": 0.5,
        "test_rows": int(len(y_test)),
        "actual_fraud_cases": int(y_test.sum()),
        "predicted_fraud_cases": int(predictions.sum()),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "precision": float(precision_score(
            y_test, predictions, zero_division=0
        )),
        "recall": float(recall_score(
            y_test, predictions, zero_division=0
        )),
        "f1_score": float(f1_score(
            y_test, predictions, zero_division=0
        )),
        "pr_auc_average_precision": float(
            average_precision_score(y_test, probabilities)
        ),
    }

    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "logistic_regression_baseline.joblib")

    report_dir = Path("reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    with open(report_dir / "baseline_metrics.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report
