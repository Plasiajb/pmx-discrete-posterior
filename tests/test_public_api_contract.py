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


def _posterior_function():
    return getattr(pmx, "compute_posterior", None) or getattr(pmx, "compute_laplace_posterior")


def _curve_function():
    return getattr(pmx, "compute_curve", None) or getattr(pmx, "build_laplace_curve_payload")


def _config_class():
    return getattr(pmx, "LaplaceConfig", None) or getattr(pmx, "OneEtaLaplaceConfig")


def _oxc_model_class():
    return (
        getattr(pmx, "OxcSingleEtaModel", None)
        or getattr(pmx, "OxcOneEtaDoseHistoryModel", None)
        or getattr(pmx, "OxcLaplacePk")
    )


def _build_model():
    model_cls = _oxc_model_class()
    return model_cls(
        dose_mg=300.0,
        interval_h=12.0,
        obs_time_h=36.0,
        weight_kg=25.0,
        steady_state_anchor_age_h=36.0,
        candidate_dose_ages_h=(12.0, 24.0),
        candidate_dose_mg=(300.0, 300.0),
    )


def _build_config():
    config_cls = _config_class()
    try:
        return config_cls(omega_variance=OMEGA_VARIANCE, residual_sd=RESIDUAL_SD)
    except TypeError:
        return config_cls(omega_var=OMEGA_VARIANCE, sigma_add=RESIDUAL_SD)


def _build_states(prior: dict[str, float] | None = None):
    if prior is None:
        prior = {scenario: 1.0 / len(SCENARIOS) for scenario in SCENARIOS}
    builder = getattr(pmx, "build_oxc_dose_history_states", None)
    if builder is not None:
        return builder(SCENARIOS, prior)
    return tuple(pmx.DiscreteState(state_id=scenario, prior=prior[scenario], label=scenario) for scenario in SCENARIOS)


def _posterior_dict(result) -> dict[str, float]:
    posterior = getattr(result, "posterior", None)
    if isinstance(posterior, dict):
        return {str(key): float(value) for key, value in posterior.items()}
    if isinstance(result, tuple) and result and isinstance(result[0], dict):
        return {str(key): float(value) for key, value in result[0].items()}
    raise AssertionError(f"Unsupported posterior result shape: {type(result)!r}")


def _map_state(result, posterior: dict[str, float]) -> str:
    for attr in ("map_state_id", "MAP", "map_scenario"):
        value = getattr(result, attr, None)
        if value:
            return str(value)
    return max(posterior, key=posterior.get)


class PublicApiContractTests(unittest.TestCase):
    def test_public_surface_has_posterior_curve_state_config_and_oxc_model_contracts(self) -> None:
        self.assertTrue(hasattr(pmx, "DiscreteState"))
        self.assertTrue(hasattr(pmx, "compute_posterior") or hasattr(pmx, "compute_laplace_posterior"))
        self.assertTrue(hasattr(pmx, "compute_curve") or hasattr(pmx, "build_laplace_curve_payload"))
        self.assertTrue(hasattr(pmx, "LaplaceConfig") or hasattr(pmx, "OneEtaLaplaceConfig"))
        self.assertTrue(
            hasattr(pmx, "OxcSingleEtaModel")
            or hasattr(pmx, "OxcOneEtaDoseHistoryModel")
            or hasattr(pmx, "OxcLaplacePk")
        )

    def test_oxc_four_state_posterior_normalizes_and_maps_truth_like_dv(self) -> None:
        model = _build_model()
        config = _build_config()
        states = _build_states()
        observation = model.prediction("omega11", 0.0)

        result = _posterior_function()(observation, states, model, config)
        posterior = _posterior_dict(result)

        self.assertEqual(set(posterior), set(SCENARIOS))
        self.assertAlmostEqual(sum(posterior.values()), 1.0, places=10)
        self.assertEqual(_map_state(result, posterior), "omega11")
        self.assertGreater(posterior["omega11"], 0.75)
        self.assertTrue(all(0.0 <= value <= 1.0 for value in posterior.values()))

        diagnostics = getattr(result, "diagnostics", {})
        if diagnostics:
            omega11 = diagnostics["omega11"]
            self.assertGreater(float(omega11.hessian), 0.0)
            self.assertTrue(math.isfinite(float(omega11.laplace_log_likelihood)))
            self.assertFalse(bool(omega11.boundary_hit))

    def test_zero_prior_structurally_excludes_state_without_breaking_normalization(self) -> None:
        model = _build_model()
        config = _build_config()
        prior = {"omega00": 0.25, "omega01": 0.25, "omega10": 0.50, "omega11": 0.0}
        states = _build_states(prior)
        observation = model.prediction("omega11", 0.0)

        result = _posterior_function()(observation, states, model, config)
        posterior = _posterior_dict(result)

        self.assertEqual(posterior["omega11"], 0.0)
        self.assertAlmostEqual(sum(posterior.values()), 1.0, places=10)
        self.assertEqual(_map_state(result, posterior), "omega10")

    def test_unnormalized_prior_normalizes_and_negative_prior_is_rejected(self) -> None:
        model = _build_model()
        config = _build_config()
        states = _build_states({"omega00": 2.0, "omega01": 2.0, "omega10": 4.0, "omega11": 0.0})

        result = _posterior_function()(model.prediction("omega10", 0.0), states, model, config)
        posterior = _posterior_dict(result)

        self.assertAlmostEqual(sum(posterior.values()), 1.0, places=10)
        bad_states = _build_states({"omega00": 0.25, "omega01": -0.1, "omega10": 0.5, "omega11": 0.35})
        with self.assertRaises(ValueError):
            _posterior_function()(1.0, bad_states, model, config)

    def test_all_zero_prior_is_rejected(self) -> None:
        model = _build_model()
        config = _build_config()
        states = _build_states({scenario: 0.0 for scenario in SCENARIOS})

        with self.assertRaises(ValueError):
            _posterior_function()(1.0, states, model, config)

    def test_vector_observation_is_explicitly_out_of_scope(self) -> None:
        model = _build_model()
        config = _build_config()
        states = _build_states()

        with self.assertRaises(NotImplementedError):
            _posterior_function()([1.0, 2.0], states, model, config)

    def test_nonpositive_hessian_marks_result_indeterminate_and_exports_warning(self) -> None:
        model = _build_model()
        config = _build_config()
        states = _build_states()

        with patch("pmx_discrete_posterior.engine.finite_second_derivative", return_value=-1.0):
            result = _posterior_function()(model.prediction("omega11", 0.0), states, model, config)
            self.assertTrue(result.indeterminate)
            self.assertTrue(any("nonpositive_or_nonfinite_hessian" in item for item in result.warnings))
            self.assertFalse(result.diagnostics["omega11"].valid_laplace)

            payload = _curve_function()([model.prediction("omega11", 0.0)], states, model, config)
            self.assertTrue(any("nonpositive_or_nonfinite_hessian" in row["warnings"] for row in payload["long_rows"]))
            self.assertTrue(any(bool(row["indeterminate"]) for row in payload["point_parameter_rows"]))

    def test_oxc_state_metadata_exposes_event_level_taken_missed_semantics(self) -> None:
        states = _build_states()
        by_state = {state.state_id: state for state in states}

        self.assertEqual(by_state["omega01"].metadata["latest_24h"], "missed")
        self.assertEqual(by_state["omega01"].metadata["second_most_recent_12h"], "given")
        self.assertIn("latest_24h=missed", by_state["omega01"].display_label)
        self.assertIn("second_most_recent_12h=given", by_state["omega01"].display_label)
        candidate_events = by_state["omega01"].metadata["candidate_events"]
        self.assertEqual(candidate_events[0]["event_label"], "latest_24h")
        self.assertEqual(candidate_events[0]["dose_state"], "missed")

    def test_curve_payload_has_gui4_compatible_pmix_and_diagnostic_rows(self) -> None:
        model = _build_model()
        config = _build_config()
        states = _build_states()
        x_grid = [
            model.prediction("omega00", 0.0),
            model.prediction("omega10", 0.0),
            model.prediction("omega11", 0.0),
        ]

        payload = _curve_function()(x_grid, states, model, config)

        self.assertEqual(len(payload["curves"]), len(SCENARIOS))
        self.assertEqual(len(payload["long_rows"]), len(x_grid) * len(SCENARIOS))
        self.assertEqual(len(payload["point_parameter_rows"]), len(x_grid) * len(SCENARIOS))

        for point_index in range(1, len(x_grid) + 1):
            rows = [row for row in payload["long_rows"] if int(row["point_index"]) == point_index]
            self.assertEqual(len(rows), len(SCENARIOS))
            self.assertAlmostEqual(sum(float(row["PMIX"]) for row in rows), 1.0, places=10)
            for row in rows:
                for key in (
                    "DV",
                    "scenario_key",
                    "posterior_probability",
                    "PMIX",
                    "eta_hat",
                    "pred_at_eta_hat",
                    "psi_hat",
                    "hessian",
                    "laplace_log_likelihood",
                    "MAP",
                    "POST_MAX",
                    "source",
                ):
                    self.assertIn(key, row)


if __name__ == "__main__":
    unittest.main()
