
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


TARGET = "isFraud"


def create_train_test_split(
    input_path="data/processed/transactions_clean.csv",
    output_dir="data/processed/splits",
    test_size=0.20,
    random_state=42,
):
    """Create a reproducible, stratified baseline split."""

    source = Path(input_path)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(source)

    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found.")

    if df[TARGET].isna().any():
        raise ValueError("Target contains missing values.")

    if not df[TARGET].isin([0, 1]).all():
        raise ValueError("Target must contain only 0 and 1.")

    # Separate model inputs from the target.
    X = df.drop(columns=[TARGET, "isFlaggedFraud"])
    y = df[TARGET]

    # Stratification preserves the rare-fraud proportion approximately.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    train = X_train.copy()
    train[TARGET] = y_train

    test = X_test.copy()
    test[TARGET] = y_test

    train_path = destination / "train.csv"
    test_path = destination / "test.csv"

    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)

    return {
        "total_rows": int(len(df)),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_fraud_count": int(y_train.sum()),
        "test_fraud_count": int(y_test.sum()),
        "train_fraud_rate_percent": round(float(y_train.mean() * 100), 5),
        "test_fraud_rate_percent": round(float(y_test.mean() * 100), 5),
        "train_path": str(train_path),
        "test_path": str(test_path),
    }
