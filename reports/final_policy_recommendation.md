# FinTech Fraud Risk Intelligence System
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
| Alerts | 20,852 | 7,104 |
| False positives | 19,600 | 5,878 |
| Missed fraud cases | 0 | 26 |
| Estimated FP cost | $98,000.00 | $29,390.00 |
| Estimated missed-fraud loss | $0.00 | $2,681,809.26 |
| Estimated total cost | $98,000.00 | $2,711,199.26 |

### 3. Cost sensitivity findings

The sensitivity analysis tested 12 combinations of:
- False-positive handling cost: $5, $25 and $100.
- Missed-fraud loss: 10%, 25%, 50% and 100% of transaction amount.

Within these tested scenarios:
- Global policy ranked first in 8 scenarios.
- Segment-specific policy ranked first in 4 scenarios.

These are scenario counts, not probabilities that a policy will be best
in real operations.

At a false-positive cost of $5, the estimated break-even realized-loss
fraction is **2.56%** of the
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
