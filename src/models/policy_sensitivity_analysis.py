
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

DATA_DIR = Path("data/processed/temporal_splits")
MODEL_PATH = Path("models/logistic_regression_temporal.joblib")
POLICY_PATH = Path("reports/risk_policy_comparison.json")
REPORT_PATH = Path("reports/policy_sensitivity_report.json")

# Business assumptions to test
FP_COSTS = [5, 25, 100]
FRAUD_LOSS_FRACTIONS = [0.10, 0.25, 0.50, 1.00]

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


def main():
    print("Loading model, policy thresholds, and test data...")

    model = joblib.load(MODEL_PATH)
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    test = pd.read_csv(DATA_DIR / "test.csv")

    test = test[
        test["type"].isin(["CASH_OUT", "TRANSFER"])
    ].copy()

    print("Scoring chronological test data...")
    test["score"] = model.predict_proba(
        prepare_features(test)
    )[:, 1]

    # Reuse thresholds selected previously on validation data.
    thresholds = {
        (item["type"], item["target_validation_capacity_pct"]):
            item["threshold"]
        for item in policy["results"]
    }

    results = []

    for capacity in [5.0, 10.0, 15.0, 20.0]:
        all_predictions = []

        for transaction_type in ["CASH_OUT", "TRANSFER"]:
            segment = test[
                test["type"] == transaction_type
            ].copy()

            threshold = thresholds[(transaction_type, capacity)]
            segment["predicted_fraud"] = (
                segment["score"] >= threshold
            )

            all_predictions.append(segment)

        evaluated = pd.concat(
            all_predictions, ignore_index=True
        )

        y = evaluated["isFraud"].to_numpy(dtype=int)
        pred = evaluated["predicted_fraud"].to_numpy(dtype=bool)
        amounts = evaluated["amount"].to_numpy(dtype=float)

        fp_mask = pred & (y == 0)
        fn_mask = (~pred) & (y == 1)

        false_positives = int(fp_mask.sum())
        missed_frauds = int(fn_mask.sum())
        alerts = int(pred.sum())

        for fp_cost in FP_COSTS:
            for loss_fraction in FRAUD_LOSS_FRACTIONS:
                investigation_cost = (
                    false_positives * fp_cost
                )

                missed_fraud_loss = float(
                    amounts[fn_mask].sum() * loss_fraction
                )

                total_cost = (
                    investigation_cost + missed_fraud_loss
                )

                results.append({
                    "target_validation_capacity_pct": capacity,
                    "fp_cost_usd": fp_cost,
                    "missed_fraud_loss_fraction": loss_fraction,
                    "test_alerts": alerts,
                    "false_positives": false_positives,
                    "missed_frauds": missed_frauds,
                    "investigation_cost_usd": investigation_cost,
                    "estimated_missed_fraud_loss_usd":
                        missed_fraud_loss,
                    "total_estimated_cost_usd": total_cost,
                })

    results_df = pd.DataFrame(results)

    # Find the lowest-cost policy for every cost scenario.
    winners = (
        results_df.sort_values("total_estimated_cost_usd")
        .groupby(
            ["fp_cost_usd", "missed_fraud_loss_fraction"],
            as_index=False,
        )
        .first()
    )

    print("\n=== ALL POLICY SCENARIOS ===")
    print(
        results_df.round(2).to_string(index=False)
    )

    print("\n=== LOWEST-COST POLICY PER SCENARIO ===")
    print(
        winners[
            [
                "fp_cost_usd",
                "missed_fraud_loss_fraction",
                "target_validation_capacity_pct",
                "test_alerts",
                "missed_frauds",
                "total_estimated_cost_usd",
            ]
        ].round(2).to_string(index=False)
    )

    report = {
        "model": "Chronological Pre-Transaction Logistic Regression",
        "test_rows": int(len(test)),
        "false_positive_cost_scenarios_usd": FP_COSTS,
        "missed_fraud_loss_fractions": FRAUD_LOSS_FRACTIONS,
        "results": results,
        "lowest_cost_policy_per_scenario":
            winners.to_dict(orient="records"),
        "warning": (
            "Illustrative assumptions, not verified bank costs. "
            "Thresholds were selected on validation data. "
            "All cost scenarios use the same historical test period."
        ),
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(f"\nReport saved: {REPORT_PATH}")


if __name__ == "__main__":
    main()
