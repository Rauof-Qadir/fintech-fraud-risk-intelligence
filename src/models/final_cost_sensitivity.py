
import json
from pathlib import Path

import pandas as pd

INPUT_PATH = Path("reports/final_policy_audit.json")
OUTPUT_PATH = Path("reports/final_cost_sensitivity.json")

FP_COSTS = [5, 25, 100]
LOSS_FRACTIONS = [0.10, 0.25, 0.50, 1.00]


def main():
    report = json.loads(INPUT_PATH.read_text(encoding="utf-8"))

    # Combine segment-level results for each policy.
    policies = {}

    for row in report["segment_results"]:
        name = row["policy"]
        policies.setdefault(
            name,
            {"false_positives": 0, "missed_fraud_amount": 0.0},
        )
        policies[name]["false_positives"] += row["false_positives"]
        policies[name]["missed_fraud_amount"] += (
            row["missed_fraud_amount_usd"]
        )

    rows = []

    for policy_name, values in policies.items():
        for fp_cost in FP_COSTS:
            for loss_fraction in LOSS_FRACTIONS:
                fp_cost_total = values["false_positives"] * fp_cost
                missed_loss = (
                    values["missed_fraud_amount"] * loss_fraction
                )

                rows.append({
                    "policy": policy_name,
                    "fp_cost_usd": fp_cost,
                    "missed_loss_fraction_pct": loss_fraction * 100,
                    "false_positives": values["false_positives"],
                    "missed_fraud_amount_usd": values[
                        "missed_fraud_amount"
                    ],
                    "estimated_fp_cost_usd": fp_cost_total,
                    "estimated_missed_loss_usd": missed_loss,
                    "estimated_total_cost_usd": fp_cost_total + missed_loss,
                })

    df = pd.DataFrame(rows)

    # Rank policies within each cost-assumption scenario.
    df["scenario_rank"] = (
        df.groupby(["fp_cost_usd", "missed_loss_fraction_pct"])[
            "estimated_total_cost_usd"
        ]
        .rank(method="min")
        .astype(int)
    )

    df = df.sort_values(
        ["fp_cost_usd", "missed_loss_fraction_pct", "scenario_rank"]
    )

    # Break-even fraction for the segment-specific policy versus global.
    global_policy = policies["global_threshold_0.19"]
    segment_policy = policies["segment_specific_20pct_target"]

    fp_savings = (
        global_policy["false_positives"]
        - segment_policy["false_positives"]
    ) * 5

    additional_missed_amount = (
        segment_policy["missed_fraud_amount"]
        - global_policy["missed_fraud_amount"]
    )

    break_even_fraction = (
        fp_savings / additional_missed_amount
        if additional_missed_amount > 0
        else None
    )

    output = {
        "scenarios": df.to_dict(orient="records"),
        "break_even_at_fp_cost_5_usd": {
            "fp_handling_savings_usd": fp_savings,
            "additional_missed_transaction_amount_usd":
                additional_missed_amount,
            "break_even_loss_fraction_pct": (
                break_even_fraction * 100
                if break_even_fraction is not None else None
            ),
        },
        "assumptions": {
            "fp_costs_usd": FP_COSTS,
            "missed_loss_fractions_pct": [
                fraction * 100 for fraction in LOSS_FRACTIONS
            ],
            "warning": (
                "Scenario estimates only. Transaction amount is not "
                "necessarily realized bank loss, and test results do "
                "not guarantee future performance."
            ),
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print("\n=== COST SENSITIVITY ===")
    print(
        df[
            [
                "fp_cost_usd",
                "missed_loss_fraction_pct",
                "policy",
                "estimated_fp_cost_usd",
                "estimated_missed_loss_usd",
                "estimated_total_cost_usd",
                "scenario_rank",
            ]
        ].round(2).to_string(index=False)
    )

    print("\n=== BREAK-EVEN ANALYSIS (FP COST = $5) ===")
    print(json.dumps(output["break_even_at_fp_cost_5_usd"], indent=2))
    print(f"\nSaved report: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
