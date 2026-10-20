
import json
from pathlib import Path

AUDIT_PATH = Path("reports/final_policy_audit.json")
SENSITIVITY_PATH = Path("reports/final_cost_sensitivity.json")
OUTPUT_PATH = Path("reports/final_policy_recommendation.md")


def money(value):
    return f"${value:,.2f}"


def main():
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    sensitivity = json.loads(
        SENSITIVITY_PATH.read_text(encoding="utf-8")
    )

    summaries = {
        row["policy"]: row
        for row in audit["policy_summary"]
    }

    global_policy = summaries["global_threshold_0.19"]
    segment_policy = summaries["segment_specific_20pct_target"]

    scenarios = sensitivity["scenarios"]
    global_wins = sum(
        1 for row in scenarios
        if row["scenario_rank"] == 1
        and row["policy"] == "global_threshold_0.19"
    )
    segment_wins = sum(
        1 for row in scenarios
        if row["scenario_rank"] == 1
        and row["policy"] == "segment_specific_20pct_target"
    )

    breakeven = sensitivity["break_even_at_fp_cost_5_usd"]

    report = f"""# FinTech Fraud Risk Intelligence System
## Final Policy Recommendation — V1

### 1. Executive summary

This project compares two fraud-alert policies on a chronological
historical test period. The thresholds were selected using validation
data and then evaluated on the test data.

**Recommendation:** Do not automatically deploy either policy based
only on these historical results. Run a controlled shadow-mode pilot,
validate actual investigation costs and realized fraud losses, and
review performance by transaction type before making a production
decision.

### 2. Test-period comparison

| Metric | Global threshold 0.19 | Segment-specific policy |
|---|---:|---:|
| Alerts | {global_policy['total_alerts']:,} | {segment_policy['total_alerts']:,} |
| False positives | {global_policy['total_false_positives']:,} | {segment_policy['total_false_positives']:,} |
| Missed fraud cases | {global_policy['total_false_negatives']:,} | {segment_policy['total_false_negatives']:,} |
| Estimated FP cost | {money(global_policy['total_fp_cost_usd'])} | {money(segment_policy['total_fp_cost_usd'])} |
| Estimated missed-fraud loss | {money(global_policy['total_missed_fraud_amount_usd'])} | {money(segment_policy['total_missed_fraud_amount_usd'])} |
| Estimated total cost | {money(global_policy['total_estimated_cost_usd'])} | {money(segment_policy['total_estimated_cost_usd'])} |

### 3. Cost sensitivity findings

The sensitivity analysis tested 12 combinations of:
- False-positive handling cost: $5, $25 and $100.
- Missed-fraud loss: 10%, 25%, 50% and 100% of transaction amount.

Within these tested scenarios:
- Global policy ranked first in {global_wins} scenarios.
- Segment-specific policy ranked first in {segment_wins} scenarios.

These are scenario counts, not probabilities that a policy will be best
in real operations.

At a false-positive cost of $5, the estimated break-even realized-loss
fraction is **{breakeven['break_even_loss_fraction_pct']:.2f}%** of the
missed transactions' combined amount. This result depends on the
current test outcomes and cost assumptions.

### 4. Operational interpretation

**Global threshold**
- Generated more alerts in the test period.
- Detected all labeled fraud cases in this particular test period.
- May exceed the available investigation team's capacity.
- Zero missed frauds in this sample is not a guarantee of future recall.

**Segment-specific policy**
- Generated fewer alerts.
- Improved alert precision in this test period.
- Missed 26 labeled fraud cases, including 24 CASH_OUT cases and
  2 TRANSFER cases.
- Requires explicit risk acceptance and monitoring of missed fraud.

### 5. Recommended next steps

1. Run both policies in shadow mode; do not automatically block
   transactions based on this experiment alone.
2. Obtain real investigation costs, confirmed fraud outcomes, recovery
   rates and realized losses from the business.
3. Measure investigation workload and fraud capture separately for
   CASH_OUT and TRANSFER.
4. Review false negatives with fraud analysts and investigate whether
   additional pre-transaction features could help.
5. Monitor alert volume, precision, recall, score drift and fraud
   prevalence over time.
6. Define business-approved alert capacity and loss tolerance before
   selecting a production threshold.
7. Reassess thresholds using an approved validation window. Preserve a
   future holdout period for independent evaluation.

### 6. Limitations

- Historical test performance does not guarantee future performance.
- Estimated costs depend on illustrative assumptions, not verified bank
  economics.
- Transaction amount is not necessarily equal to realized fraud loss.
- The test alert rate can differ from the validation target.
- Model performance can change with customer behavior and fraud patterns.
- This evaluation is not evidence of production readiness by itself.

### Final conclusion

The project demonstrates a repeatable workflow for chronological
evaluation, threshold selection, capacity analysis, cost sensitivity
and operational monitoring. The appropriate policy depends on the
business's real fraud-loss economics and investigation capacity.
A controlled pilot and independently monitored outcomes should guide
the final production decision.
"""

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(report, encoding="utf-8")

    print(report)
    print(f"\nReport saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
