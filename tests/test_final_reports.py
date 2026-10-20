
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def load_json(filename):
    path = REPORTS / filename
    assert path.exists(), f"Missing report: {path}"
    return json.loads(path.read_text(encoding="utf-8"))


def test_final_audit_contains_both_policies():
    report = load_json("final_policy_audit.json")

    policy_names = {
        row["policy"] for row in report["policy_summary"]
    }

    assert "global_threshold_0.19" in policy_names
    assert "segment_specific_20pct_target" in policy_names


def test_policy_summary_costs_are_consistent():
    report = load_json("final_policy_audit.json")

    for row in report["policy_summary"]:
        expected = (
            row["total_fp_cost_usd"]
            + row["total_missed_fraud_amount_usd"]
        )

        assert abs(
            row["total_estimated_cost_usd"] - expected
        ) < 0.01


def test_segment_results_have_valid_confusion_counts():
    report = load_json("final_policy_audit.json")

    for row in report["segment_results"]:
        assert row["alerts"] == (
            row["true_positives"] + row["false_positives"]
        )

        assert row["actual_frauds"] == (
            row["true_positives"] + row["false_negatives"]
        )

        assert 0 <= row["precision_pct"] <= 100
        assert row["recall_pct"] is None or (
            0 <= row["recall_pct"] <= 100
        )


def test_sensitivity_break_even_is_valid():
    report = load_json("final_cost_sensitivity.json")

    break_even = report["break_even_at_fp_cost_5_usd"][
        "break_even_loss_fraction_pct"
    ]

    assert break_even is not None
    assert 0 < break_even < 100
    assert len(report["scenarios"]) == 24


def test_readme_links_to_final_recommendation():
    readme_path = ROOT / "README.md"
    recommendation_path = REPORTS / "final_policy_recommendation.md"

    assert readme_path.exists(), "README.md is missing"
    assert recommendation_path.exists(), (
        "Run the final policy recommendation script first"
    )

    readme = readme_path.read_text(encoding="utf-8")

    assert "reports/final_policy_recommendation.md" in readme
