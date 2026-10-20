
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def run_eda(csv_path: str, output_dir: str = "reports/figures") -> dict:
    """Explore transaction patterns and save reproducible EDA reports."""

    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_file.resolve()}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_file)

    required = {"type", "amount", "isFraud", "isFlaggedFraud"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Required columns missing: {sorted(missing)}")

    # 1. Transaction type counts
    type_counts = df["type"].value_counts()

    plt.figure(figsize=(9, 5))
    type_counts.plot(kind="bar")
    plt.title("Transaction Volume by Type")
    plt.xlabel("Transaction Type")
    plt.ylabel("Number of Transactions")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_path / "transaction_type_counts.png", dpi=150)
    plt.close()

    # 2. Fraud rate by transaction type
    fraud_by_type = (
        df.groupby("type", observed=True)["isFraud"]
        .agg(transactions="count", fraud_count="sum", fraud_rate="mean")
        .reset_index()
    )
    fraud_by_type["fraud_rate_percent"] = fraud_by_type["fraud_rate"] * 100

    plt.figure(figsize=(9, 5))
    plt.bar(
        fraud_by_type["type"],
        fraud_by_type["fraud_rate_percent"],
    )
    plt.title("Fraud Rate by Transaction Type")
    plt.xlabel("Transaction Type")
    plt.ylabel("Fraud Rate (%)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_path / "fraud_rate_by_type.png", dpi=150)
    plt.close()

    # 3. Transaction amount distribution
    plt.figure(figsize=(9, 5))
    plt.hist(df["amount"], bins=100)
    plt.title("Transaction Amount Distribution")
    plt.xlabel("Transaction Amount")
    plt.ylabel("Number of Transactions")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(output_path / "amount_distribution.png", dpi=150)
    plt.close()

    # 4. Compare amounts for fraud and non-fraud transactions.
    # A sample makes the chart easier to read and limits plotting work.
    fraud_amounts = df.loc[df["isFraud"] == 1, "amount"]
    normal_amounts = df.loc[df["isFraud"] == 0, "amount"]

    plt.figure(figsize=(9, 5))
    plt.hist(
        normal_amounts.sample(n=min(50000, len(normal_amounts)), random_state=42),
        bins=60,
        alpha=0.65,
        label="Non-fraud sample",
    )
    plt.hist(
        fraud_amounts,
        bins=60,
        alpha=0.65,
        label="Fraud",
    )
    plt.title("Transaction Amounts: Fraud vs Non-Fraud")
    plt.xlabel("Transaction Amount")
    plt.ylabel("Number of Transactions")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path / "fraud_vs_nonfraud_amount.png", dpi=150)
    plt.close()

    # Investigate zero-amount transactions without deleting them.
    zero_amounts = df.loc[df["amount"] == 0]

    report = {
        "total_transactions": int(len(df)),
        "fraud_transactions": int(df["isFraud"].sum()),
        "fraud_rate_percent": round(float(df["isFraud"].mean() * 100), 5),
        "median_amount": float(df["amount"].median()),
        "mean_amount": float(df["amount"].mean()),
        "maximum_amount": float(df["amount"].max()),
        "zero_amount_transactions": int(len(zero_amounts)),
        "zero_amount_fraud_transactions": int(zero_amounts["isFraud"].sum()),
        "fraud_by_type": fraud_by_type.to_dict(orient="records"),
        "charts_directory": str(output_path),
    }

    with open(output_path.parent / "eda_report.json", "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    return report
