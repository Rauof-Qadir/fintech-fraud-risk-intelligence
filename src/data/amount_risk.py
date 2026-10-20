
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def analyze_amount_risk(
    csv_path="data/raw/transactions.csv",
    output_dir="reports/figures",
):
    df = pd.read_csv(csv_path, usecols=["amount", "isFraud"])

    # Keep zero amounts separate so we can investigate them.
    bins = [
        -0.001, 0, 1, 1_000, 10_000, 50_000,
        100_000, 500_000, 1_000_000, 5_000_000, float("inf")
    ]

    labels = [
        "Zero",
        "(0, 1]",
        "(1, 1K]",
        "(1K, 10K]",
        "(10K, 50K]",
        "(50K, 100K]",
        "(100K, 500K]",
        "(500K, 1M]",
        "(1M, 5M]",
        "Above 5M",
    ]

    df["amount_band"] = pd.cut(
        df["amount"],
        bins=bins,
        labels=labels,
        include_lowest=True,
    )

    report = (
        df.groupby("amount_band", observed=False)
        .agg(
            transactions=("isFraud", "size"),
            fraud_transactions=("isFraud", "sum"),
            fraud_rate=("isFraud", "mean"),
        )
        .reset_index()
    )

    report["fraud_rate_percent"] = report["fraud_rate"] * 100

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    report.to_csv("reports/amount_risk_report.csv", index=False)

    # A zero transaction count means the band has no observations.
    chart_data = report[report["transactions"] > 0]

    plt.figure(figsize=(11, 6))
    plt.bar(
        chart_data["amount_band"].astype(str),
        chart_data["fraud_rate_percent"],
    )
    plt.title("Fraud Rate by Transaction Amount Range")
    plt.xlabel("Transaction Amount Range")
    plt.ylabel("Fraud Rate (%)")
    plt.xticks(rotation=40, ha="right")
    plt.tight_layout()
    plt.savefig(output / "fraud_rate_by_amount_band.png", dpi=150)
    plt.close()

    return report


if __name__ == "__main__":
    result = analyze_amount_risk()
    print(result.to_string(index=False))
