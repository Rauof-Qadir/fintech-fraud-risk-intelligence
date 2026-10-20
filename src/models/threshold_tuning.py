
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    average_precision_score,
)


FEATURES = ["amount", "type"]
TARGET = "isFraud"


def make_model():
    preprocessing = ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), ["amount"]),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]), ["type"]),
    ])

    return Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        )),
    ])


def tune_threshold(
    train_path="data/processed/splits/train.csv",
    test_path="data/processed/splits/test.csv",
    target_recall=0.80,
):
    columns = FEATURES + [TARGET]

    train = pd.read_csv(train_path, usecols=columns)
    test = pd.read_csv(test_path, usecols=columns)

    # Reserve part of training data for threshold selection.
    fit_data, validation_data = train_test_split(
        train,
        test_size=0.20,
        random_state=42,
        stratify=train[TARGET],
    )

    X_fit = fit_data[FEATURES]
    y_fit = fit_data[TARGET]
    X_val = validation_data[FEATURES]
    y_val = validation_data[TARGET]

    validation_model = make_model()
    validation_model.fit(X_fit, y_fit)

    val_probabilities = validation_model.predict_proba(X_val)[:, 1]

    precision, recall, thresholds = precision_recall_curve(
        y_val, val_probabilities
    )

    # Each threshold corresponds to precision[i] and recall[i].
    eligible = np.flatnonzero(recall[:-1] >= target_recall)

    if len(eligible) == 0:
        raise RuntimeError(
            "No threshold meets the requested validation recall."
        )

    # Among eligible thresholds, select the highest precision.
    best_index = eligible[np.argmax(precision[eligible])]
    threshold = float(thresholds[best_index])

    validation_predictions = (
        val_probabilities >= threshold
    ).astype("int8")

    # Refit on all training data; never tune using the test labels.
    final_model = make_model()
    final_model.fit(train[FEATURES], train[TARGET])

    test_probabilities = final_model.predict_proba(test[FEATURES])[:, 1]
    test_predictions = (test_probabilities >= threshold).astype("int8")

    tn, fp, fn, tp = confusion_matrix(
        test[TARGET], test_predictions, labels=[0, 1]
    ).ravel()

    report = {
        "selected_threshold": threshold,
        "selection_method": (
            f"Highest validation precision with recall >= {target_recall:.0%}"
        ),
        "validation_precision": float(
            precision_score(y_val, validation_predictions, zero_division=0)
        ),
        "validation_recall": float(
            recall_score(y_val, validation_predictions, zero_division=0)
        ),
        "test_precision": float(
            precision_score(test[TARGET], test_predictions, zero_division=0)
        ),
        "test_recall": float(
            recall_score(test[TARGET], test_predictions, zero_division=0)
        ),
        "test_f1": float(
            f1_score(test[TARGET], test_predictions, zero_division=0)
        ),
        "test_average_precision": float(
            average_precision_score(test[TARGET], test_probabilities)
        ),
        "test_true_positives": int(tp),
        "test_false_positives": int(fp),
        "test_true_negatives": int(tn),
        "test_false_negatives": int(fn),
        "test_alert_count": int(test_predictions.sum()),
    }

    Path("models").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)

    joblib.dump(final_model, "models/logistic_regression_tuned.joblib")

    with open("reports/threshold_tuning_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


if __name__ == "__main__":
    print(json.dumps(tune_threshold(), indent=2))
