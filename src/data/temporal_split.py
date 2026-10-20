
from pathlib import Path

import pandas as pd


SOURCE = Path("data/processed/transactions_clean.csv")
OUTPUT_DIR = Path("data/processed/temporal_splits")
CHUNK_SIZE = 250_000

SPLITS = {
    "train": (1, 520),
    "validation": (521, 631),
    "test": (632, 743),
}


def main():
    if not SOURCE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {SOURCE}. "
            "Check that the cleaned CSV exists."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Remove old split files so rerunning won't duplicate rows.
    for name in SPLITS:
        path = OUTPUT_DIR / f"{name}.csv"
        if path.exists():
            path.unlink()

    row_counts = {name: 0 for name in SPLITS}
    fraud_counts = {name: 0 for name in SPLITS}
    first_steps = {name: None for name in SPLITS}
    last_steps = {name: None for name in SPLITS}

    print(f"Reading dataset in chunks: {SOURCE}")

    for chunk_number, chunk in enumerate(
        pd.read_csv(SOURCE, chunksize=CHUNK_SIZE),
        start=1,
    ):
        required = {"step", "isFraud"}
        missing = required - set(chunk.columns)

        if missing:
            raise ValueError(
                f"Dataset is missing columns: {sorted(missing)}"
            )

        for name, (start_step, end_step) in SPLITS.items():
            mask = chunk["step"].between(start_step, end_step)
            subset = chunk.loc[mask]

            if subset.empty:
                continue

            output_path = OUTPUT_DIR / f"{name}.csv"

            subset.to_csv(
                output_path,
                mode="a",
                header=not output_path.exists(),
                index=False,
            )

            row_counts[name] += len(subset)
            fraud_counts[name] += int(subset["isFraud"].sum())

            minimum = int(subset["step"].min())
            maximum = int(subset["step"].max())

            if first_steps[name] is None:
                first_steps[name] = minimum
            else:
                first_steps[name] = min(first_steps[name], minimum)

            if last_steps[name] is None:
                last_steps[name] = maximum
            else:
                last_steps[name] = max(last_steps[name], maximum)

        print(f"Processed chunk {chunk_number}")

    print("\n=== CHRONOLOGICAL SPLIT SUMMARY ===")

    for name, (start, end) in SPLITS.items():
        rows = row_counts[name]
        frauds = fraud_counts[name]
        fraud_rate = (frauds / rows * 100) if rows else 0

        print(f"\n{name.upper()}")
        print(f"Expected steps: {start}-{end}")
        print(f"Actual step range: {first_steps[name]}-{last_steps[name]}")
        print(f"Rows: {rows:,}")
        print(f"Fraud cases: {frauds:,}")
        print(f"Fraud rate: {fraud_rate:.4f}%")
        print(f"Saved: {OUTPUT_DIR / (name + '.csv')}")

    total_rows = sum(row_counts.values())
    total_frauds = sum(fraud_counts.values())

    print("\n=== TOTAL ===")
    print(f"Rows assigned to splits: {total_rows:,}")
    print(f"Fraud cases assigned: {total_frauds:,}")
    print("Split creation complete.")


if __name__ == "__main__":
    main()
