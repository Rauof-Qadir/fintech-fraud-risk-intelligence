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
from sklearn.metrics import confusion_matrix, precision_recall_curve

FEATURES = ["amount", "type"]
TARGET = "isFraud"

# Business assumption: $5 investigation cost per false-positive alert.
FALSE_POSITIVE_COST = 5.0


def make_model():
    """Create the same Logistic Regression pipeline as our baseline."""
    preprocessing = ColumnTransformer([
        (
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            ["amount"],
        ),
        (
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]),
            ["type"],
        ),
    ])

    return Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        )),
    ])


def calculate_cost(y_true, amounts, probabilities, threshold):
    """Calculate confusion-matrix counts and estimated business cost."""
    y_true = np.asarray(y_true, dtype=np.int8)
    amounts = np.asarray(amounts, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)

    predictions = (probabilities >= threshold).astype(np.int8)

    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    # False positives: investigation cost.
    fp_cost = float(fp * FALSE_POSITIVE_COST)

    # False negatives: sum of transaction amounts for missed fraud.
    missed_fraud_loss = float(
        amounts[(y_true == 1) & (predictions == 0)].sum()
    )

    total_cost = fp_cost + missed_fraud_loss

    return {
        "threshold": float(threshold),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "alerts": int(predictions.sum()),
        "precision": float(tp / (tp + fp)) if (tp + fp) else 0.0,
        "recall": float(tp / (tp + fn)) if (tp + fn) else 0.0,
        "false_positive_cost_usd": fp_cost,
        "missed_fraud_loss_usd": missed_fraud_loss,
        "total_estimated_cost_usd": float(total_cost),
    }


def optimize_business_cost(
    train_path="data/processed/splits/train.csv",
    test_path="data/processed/splits/test.csv",
):
    columns = ["amount", "type", TARGET]

    print("1/6 Loading training and test data...")
    train = pd.read_csv(train_path, usecols=columns)
    test = pd.read_csv(test_path, usecols=columns)

    # Only training data is divided into fitting and validation sets.
    print("2/6 Creating fit and validation sets...")
    fit_data, validation_data = train_test_split(
        train,
        test_size=0.20,
        random_state=42,
        stratify=train[TARGET],
    )

    print("3/6 Training model for validation...")
    validation_model = make_model()
    validation_model.fit(
        fit_data[FEATURES],
        fit_data[TARGET],
    )

    validation_probabilities = validation_model.predict_proba(
        validation_data[FEATURES]
    )[:, 1]

    y_val = validation_data[TARGET].to_numpy(dtype=np.int8)
    amount_val = validation_data["amount"].to_numpy(dtype=float)

    # Generate candidate thresholds from validation predictions only.
    _, _, candidate_thresholds = precision_recall_curve(
        y_val, validation_probabilities
    )
    candidate_thresholds = np.unique(
        np.concatenate(([0.0], candidate_thresholds, [1.0]))
    )

    # Efficiently calculate counts and missed-fraud amounts for every
    # threshold, without repeatedly scanning the entire validation set.
    print("4/6 Optimizing threshold using validation business cost...")
    order = np.argsort(-validation_probabilities)
    sorted_scores = validation_probabilities[order]
    sorted_y = y_val[order]
    sorted_amounts = amount_val[order]

    # Prefix arrays include an initial zero, so k=0 works correctly.
    prefix_tp = np.concatenate(([0], np.cumsum(sorted_y, dtype=np.int64)))
    fraud_amounts = np.where(sorted_y == 1, sorted_amounts, 0.0)
    prefix_captured_fraud_amount = np.concatenate(
        ([0.0], np.cumsum(fraud_amounts, dtype=float))
    )

    # Number of transactions whose predicted probability >= each threshold.
    k_values = np.searchsorted(
        -sorted_scores, -candidate_thresholds, side="right"
    )

    tp_values = prefix_tp[k_values]
    alert_values = k_values
    fp_values = alert_values - tp_values

    total_fraud_amount = float(amount_val[y_val == 1].sum())
    missed_loss_values = (
        total_fraud_amount - prefix_captured_fraud_amount[k_values]
    )
    fp_cost_values = fp_values * FALSE_POSITIVE_COST
    total_cost_values = fp_cost_values + missed_loss_values

    # Lowest estimated cost wins. If costs tie, prefer fewer alerts.
    best_index = int(np.lexsort(
        (alert_values, total_cost_values)
    )[0])
    best_threshold = float(candidate_thresholds[best_index])

    validation_best = calculate_cost(
        y_val, amount_val, validation_probabilities, best_threshold
    )

    validation_default = calculate_cost(
        y_val, amount_val, validation_probabilities, 0.5
    )

    # Now fit the final model on all training data.
    # The test set has not been used to select the threshold.
    print("5/6 Fitting final model and evaluating untouched test data...")
    final_model = make_model()
    final_model.fit(train[FEATURES], train[TARGET])

    test_probabilities = final_model.predict_proba(
        test[FEATURES]
    )[:, 1]

    y_test = test[TARGET].to_numpy(dtype=np.int8)
    amount_test = test["amount"].to_numpy(dtype=float)

    test_best = calculate_cost(
        y_test, amount_test, test_probabilities, best_threshold
    )
    test_default = calculate_cost(
        y_test, amount_test, test_probabilities, 0.5
    )

    savings = (
        test_default["total_estimated_cost_usd"]
        - test_best["total_estimated_cost_usd"]
    )

    report = {
        "business_assumptions": {
            "false_positive_investigation_cost_usd": FALSE_POSITIVE_COST,
            "missed_fraud_loss": "Full transaction amount",
            "note": (
                "Illustrative assumptions, not verified bank costs. "
                "Actual loss may differ from transaction amount."
            ),
        },
        "threshold_selection": {
            "method": "Minimum estimated total cost on validation data",
            "selected_threshold": best_threshold,
            "validation_optimized": validation_best,
            "validation_at_threshold_0_5": validation_default,
        },
        "untouched_test_evaluation": {
            "optimized_threshold": test_best,
            "threshold_0_5": test_default,
            "estimated_savings_vs_threshold_0_5_usd": float(savings),
        },
    }

    Path("models").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)

    joblib.dump(final_model, "models/logistic_regression_cost_optimized.joblib")

    with open(
        "reports/business_cost_report.json", "w", encoding="utf-8"
    ) as file:
        json.dump(report, file, indent=2)

    # Save validation cost curve so we can inspect alternatives later.
    cost_curve = pd.DataFrame({
        "threshold": candidate_thresholds,
        "validation_alerts": alert_values,
        "validation_false_positives": fp_values,
        "validation_true_positives": tp_values,
        "validation_missed_fraud_loss_usd": missed_loss_values,
        "validation_false_positive_cost_usd": fp_cost_values,
        "validation_total_cost_usd": total_cost_values,
    })
    cost_curve.to_csv("reports/validation_cost_curve.csv", index=False)

    print("6/6 Saved model and reports.")
    return report


if __name__ == "__main__":
    result = optimize_business_cost()
    print(json.dumps(result, indent=2))