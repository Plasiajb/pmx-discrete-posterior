from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import pmx_discrete_posterior as pmx  # noqa: E402


SCENARIOS = ("omega00", "omega01", "omega10", "omega11")
OMEGA_VARIANCE = 0.012245714245884425
RESIDUAL_SD = 1e-6


def _build_model():
    return pmx.OxcOneEtaDoseHistoryModel(
        dose_mg=300.0,
        interval_h=12.0,
        obs_time_h=36.0,
        weight_kg=25.0,
        steady_state_anchor_age_h=36.0,
        candidate_dose_ages_h=(12.0, 24.0),
        candidate_dose_mg=(300.0, 300.0),
    )


def _build_config():
    return pmx.OneEtaLaplaceConfig(omega_variance=OMEGA_VARIANCE, residual_sd=RESIDUAL_SD)


def _build_states(prior: dict[str, float] | None = None):
    if prior is None:
        prior = {scenario: 1.0 / len(SCENARIOS) for scenario in SCENARIOS}
    return pmx.build_oxc_dose_history_states(SCENARIOS, prior)


def _truth_like_result():
    model = _build_model()
    states = _build_states()
    return pmx.compute_laplace_posterior(
        model.prediction("omega11", 0.0),
        states,
        model,
        _build_config(),
    )


class ReportingLayerTests(unittest.TestCase):
    def test_public_reporting_api_is_exported(self) -> None:
        for name in (
            "NoCallRule",
            "PosteriorEvidenceReport",
            "summarize_posterior_evidence",
            "apply_no_call_rule",
            "posterior_to_long_rows",
            "posterior_to_wide_row",
            "build_case_card",
            "threshold_sensitivity_panel",
            "reweight_prior_sensitivity",
            "build_provenance_manifest",
        ):
            self.assertTrue(hasattr(pmx, name), name)

    def test_entropy_margin_and_default_no_call_rule(self) -> None:
        self.assertAlmostEqual(
            pmx.normalized_entropy({scenario: 0.25 for scenario in SCENARIOS}),
            1.0,
            places=12,
        )
        pmax, map_state, second, second_state = pmx.posterior_margin(
            {"a": 0.55, "b": 0.35, "c": 0.10}
        )
        self.assertEqual((pmax, map_state, second, second_state), (0.55, "a", 0.35, "b"))

        no_call, reasons = pmx.apply_no_call_rule(margin=0.05, entropy=0.20)
        self.assertTrue(no_call)
        self.assertIn("low_margin", reasons)

        no_call, reasons = pmx.apply_no_call_rule(margin=0.20, entropy=0.95)
        self.assertTrue(no_call)
        self.assertIn("high_entropy", reasons)

        no_call, reasons = pmx.apply_no_call_rule(margin=0.20, entropy=0.20)
        self.assertFalse(no_call)
        self.assertEqual(reasons, ())

    def test_summarize_posterior_evidence_exports_report_rows_and_safe_case_card(self) -> None:
        report = pmx.summarize_posterior_evidence(
            _truth_like_result(),
            case_metadata={"case_id": "clear_oxc_case"},
        )

        self.assertEqual(report.map_state_id, "omega11")
        self.assertFalse(report.no_call)
        self.assertGreater(report.margin, 0.10)
        self.assertLess(report.normalized_entropy, 0.80)
        self.assertEqual(len(report.state_rows), 4)

        long_rows = pmx.posterior_to_long_rows(report)
        wide_row = pmx.posterior_to_wide_row(report)
        self.assertEqual(len(long_rows), 4)
        self.assertEqual(wide_row["case_case_id"], "clear_oxc_case")
        self.assertAlmostEqual(
            sum(row["posterior_probability"] for row in long_rows),
            1.0,
            places=10,
        )
        self.assertIn("posterior_omega11", wide_row)

        card = pmx.build_case_card(report)
        self.assertIn("posterior_vector", card)
        self.assertIn("model-conditioned support", card["boundary"])
        unsafe_words = ("diagnosed", "diagnosis of", "adherence proof", "dose should")
        self.assertFalse(any(word in card["interpretation"].lower() for word in unsafe_words))

    def test_indeterminate_warnings_propagate_to_no_call(self) -> None:
        with patch("pmx_discrete_posterior.engine.finite_second_derivative", return_value=-1.0):
            report = pmx.summarize_posterior_evidence(_truth_like_result())

        self.assertTrue(report.indeterminate)
        self.assertTrue(report.no_call)
        self.assertIn("indeterminate_laplace", report.no_call_reasons)
        self.assertIn("numerical_warning", report.no_call_reasons)
        self.assertGreater(report.warnings, ())

    def test_threshold_sensitivity_panel_counts_reclassified_reports(self) -> None:
        report = pmx.summarize_posterior_evidence(_truth_like_result())
        strict_rule = pmx.NoCallRule(
            rule_id="strict_margin",
            margin_threshold=0.999999,
            entropy_threshold=0.80,
        )
        rows = pmx.threshold_sensitivity_panel([report], [pmx.DEFAULT_NO_CALL_RULE, strict_rule])

        self.assertEqual(rows[0]["rule_id"], "NC_DEFAULT")
        self.assertEqual(rows[0]["denominator"], 1)
        self.assertEqual(rows[0]["no_call"], 0)
        self.assertEqual(rows[1]["rule_id"], "strict_margin")
        self.assertEqual(rows[1]["no_call"], 1)

    def test_prior_reweighting_uses_stored_log_likelihoods(self) -> None:
        report = pmx.summarize_posterior_evidence(_truth_like_result())
        reweighted = pmx.reweight_prior_sensitivity(
            report,
            {
                "zero_truth_state": {
                    "omega00": 1.0,
                    "omega01": 1.0,
                    "omega10": 1.0,
                    "omega11": 0.0,
                }
            },
        )

        self.assertEqual(len(reweighted), 1)
        shifted = reweighted[0]
        self.assertEqual(shifted.case_metadata["prior_id"], "zero_truth_state")
        self.assertEqual(shifted.posterior["omega11"], 0.0)
        self.assertAlmostEqual(sum(shifted.posterior.values()), 1.0, places=10)
        self.assertTrue(math.isfinite(float(shifted.log_evidence)))

    def test_provenance_manifest_records_no_call_rule(self) -> None:
        manifest = pmx.build_provenance_manifest(
            analysis_id="phase144_reporting_layer_test",
            package_version=pmx.__version__,
            model_id="oxc_demo",
            state_order=SCENARIOS,
            prior_id="equal",
        )

        self.assertEqual(manifest["analysis_id"], "phase144_reporting_layer_test")
        self.assertEqual(manifest["no_call_rule"]["rule_id"], "NC_DEFAULT")
        self.assertEqual(manifest["state_order"], list(SCENARIOS))


if __name__ == "__main__":
    unittest.main()
