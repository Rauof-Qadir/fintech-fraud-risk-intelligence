
from pathlib import Path

import pandas as pd


NUMERIC_COLUMNS = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud",
]

REQUIRED_COLUMNS = [
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud",
]


def clean_dataset(
    input_path="data/raw/transactions.csv",
    output_path="data/processed/transactions_clean.csv",
    chunksize=500_000,
):
    """Validate and standardize transactions without silently dropping rows."""

    source = Path(input_path)
    destination = Path(output_path)

    if not source.exists():
        raise FileNotFoundError(f"Input file not found: {source.resolve()}")

    destination.parent.mkdir(parents=True, exist_ok=True)

    # Check the schema before processing millions of rows.
    header = pd.read_csv(source, nrows=0)
    missing_columns = set(REQUIRED_COLUMNS) - set(header.columns)

    if missing_columns:
        raise ValueError(
            f"Required columns missing: {sorted(missing_columns)}"
        )

    if list(header.columns) != REQUIRED_COLUMNS:
        raise ValueError(
            "Column names or order differ from the expected source schema."
        )

    total_rows = 0
    zero_amount_rows = 0
    first_chunk = True

    for chunk in pd.read_csv(
        source,
        chunksize=chunksize,
        dtype={"type": "string", "nameOrig": "string", "nameDest": "string"},
    ):
        # Convert numeric fields consistently.
        for column in NUMERIC_COLUMNS:
            chunk[column] = pd.to_numeric(chunk[column], errors="raise")

        if chunk[REQUIRED_COLUMNS].isna().any().any():
            raise ValueError(
                f"Missing values detected near row {total_rows}."
            )

        if (chunk["amount"] < 0).any():
            raise ValueError("Negative transaction amounts detected.")

        if (chunk["step"] <= 0).any():
            raise ValueError("Non-positive transaction steps detected.")

        for column in ["isFraud", "isFlaggedFraud"]:
            if not chunk[column].isin([0, 1]).all():
                raise ValueError(f"Invalid binary labels in {column}.")

        if (
            (chunk["oldbalanceOrg"] < 0).any()
            or (chunk["newbalanceOrig"] < 0).any()
            or (chunk["oldbalanceDest"] < 0).any()
            or (chunk["newbalanceDest"] < 0).any()
        ):
            raise ValueError("Negative account balances detected.")

        zero_amount_rows += int((chunk["amount"] == 0).sum())
        total_rows += len(chunk)

        # Preserve every row, including unusual zero-amount transactions.
        chunk.to_csv(
            destination,
            index=False,
            mode="w" if first_chunk else "a",
            header=first_chunk,
        )
        first_chunk = False

    return {
        "input_file": str(source),
        "output_file": str(destination),
        "rows_processed": total_rows,
        "zero_amount_rows_preserved": zero_amount_rows,
        "cleaning_policy": "Validate and preserve; do not silently drop rows.",
    }
