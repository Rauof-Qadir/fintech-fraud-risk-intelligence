
from pathlib import Path

import pandas as pd


def inspect_dataset(csv_path: str) -> dict:
    """Profile data quality and basic fraud patterns."""

    path = Path(csv_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path.resolve()}"
        )

    df = pd.read_csv(path)

    required = {
        "step", "type", "amount",
        "oldbalanceOrg", "newbalanceOrig",
        "oldbalanceDest", "newbalanceDest",
        "isFraud", "isFlaggedFraud",
    }

    missing_columns = sorted(required - set(df.columns))

    if missing_columns:
        raise ValueError(
            f"Required columns missing: {missing_columns}"
        )

    fraud_counts = df["isFraud"].value_counts().to_dict()

    fraud_by_type = (
        df.groupby("type", observed=True)["isFraud"]
        .agg(["count", "sum", "mean"])
        .rename(columns={
            "count": "transactions",
            "sum": "fraud_transactions",
            "mean": "fraud_rate",
        })
        .reset_index()
    )

    report = {
        "file": path.name,
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_cells": int(df.isna().sum().sum()),

        "column_names": df.columns.tolist(),

        "data_types": {
            col: str(dtype)
            for col, dtype in df.dtypes.items()
        },

        "missing_by_column": {
            col: int(count)
            for col, count in df.isna().sum().items()
        },

        "business_quality_checks": {
            "negative_amounts": int(
                (df["amount"] < 0).sum()
            ),
            "zero_amounts": int(
                (df["amount"] == 0).sum()
            ),
            "negative_sender_balances": int(
                (
                    (df["oldbalanceOrg"] < 0)
                    | (df["newbalanceOrig"] < 0)
                ).sum()
            ),
            "negative_receiver_balances": int(
                (
                    (df["oldbalanceDest"] < 0)
                    | (df["newbalanceDest"] < 0)
                ).sum()
            ),
            "non_positive_steps": int(
                (df["step"] <= 0).sum()
            ),
            "invalid_fraud_labels": int(
                (~df["isFraud"].isin([0, 1])).sum()
            ),
            "invalid_flag_labels": int(
                (~df["isFlaggedFraud"].isin([0, 1])).sum()
            ),
        },

        "fraud_summary": {
            "class_counts": {
                str(key): int(value)
                for key, value in fraud_counts.items()
            },
            "fraud_rate": float(df["isFraud"].mean()),
            "flagged_transactions": int(
                df["isFlaggedFraud"].sum()
            ),
        },

        "transaction_types": fraud_by_type.to_dict(
            orient="records"
        ),
    }

    return report
