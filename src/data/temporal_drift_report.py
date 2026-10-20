
from pathlib import Path

import pandas as pd


DATA_DIR = Path("data/processed/temporal_splits")

FILES = {
    "train": DATA_DIR / "train.csv",
    "validation": DATA_DIR / "validation.csv",
    "test": DATA_DIR / "test.csv",
}


def main():
    summaries = []
    type_summaries = []

    for split_name, path in FILES.items():
        print(f"Analyzing {split_name}...")

        df = pd.read_csv(
            path,
            usecols=[
                "step",
                "type",
                "amount",
                "oldbalanceOrg",
                "oldbalanceDest",
                "isFraud",
            ],
        )

        summaries.append({
            "split": split_name,
            "rows": len(df),
            "first_step": int(df["step"].min()),
            "last_step": int(df["step"].max()),
            "fraud_count": int(df["isFraud"].sum()),
            "fraud_rate_pct": float(
                df["isFraud"].mean() * 100
            ),
            "median_amount": float(df["amount"].median()),
            "mean_amount": float(df["amount"].mean()),
            "median_sender_balance": float(
                df["oldbalanceOrg"].median()
            ),
            "median_receiver_balance": float(
                df["oldbalanceDest"].median()
            ),
        })

        grouped = (
            df.groupby("type", observed=True)
            .agg(
                transactions=("isFraud", "size"),
                frauds=("isFraud", "sum"),
                median_amount=("amount", "median"),
            )
            .reset_index()
        )

        grouped["split"] = split_name
        grouped["fraud_rate_pct"] = (
            grouped["frauds"] / grouped["transactions"] * 100
        )

        type_summaries.append(grouped)

    summary_df = pd.DataFrame(summaries)
    type_df = pd.concat(type_summaries, ignore_index=True)

    output_dir = Path("reports")
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_df.to_csv(
        output_dir / "temporal_drift_summary.csv",
        index=False,
    )
    type_df.to_csv(
        output_dir / "temporal_drift_by_type.csv",
        index=False,
    )

    print("\n=== OVERALL TEMPORAL DRIFT ===")
    print(summary_df.round(4).to_string(index=False))

    print("\n=== FRAUD RATE BY TYPE AND PERIOD ===")
    print(
        type_df[
            [
                "split",
                "type",
                "transactions",
                "frauds",
                "fraud_rate_pct",
                "median_amount",
            ]
        ].round(4).to_string(index=False)
    )

    print("\nReports saved:")
    print("reports/temporal_drift_summary.csv")
    print("reports/temporal_drift_by_type.csv")


if __name__ == "__main__":
    main()
