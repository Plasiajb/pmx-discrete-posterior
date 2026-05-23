from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
EXAMPLE_PATH = PROJECT_ROOT / "examples" / "multieta_stress_test.py"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import pmx_discrete_posterior as pmx  # noqa: E402


def _load_stress_module():
    spec = importlib.util.spec_from_file_location("multieta_stress_test", EXAMPLE_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Could not load example module from {EXAMPLE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class MultiEtaStressApiTests(unittest.TestCase):
    def test_multieta_stress_laplace_matches_gauss_hermite_reference(self) -> None:
        module = _load_stress_module()
        result = module.build_stress_test(gh_nodes=31)

        self.assertTrue(result["map_agreement"])
        self.assertEqual(result["laplace_map"], "state11")
        self.assertEqual(result["quadrature_map"], "state11")
        self.assertLess(result["max_abs_posterior_diff"], 0.005)
        self.assertFalse(result["laplace_indeterminate"])
        self.assertAlmostEqual(sum(float(row["laplace_posterior"]) for row in result["rows"]), 1.0, places=10)
        self.assertAlmostEqual(sum(float(row["quadrature_posterior"]) for row in result["rows"]), 1.0, places=10)
        self.assertTrue(all(bool(row["valid_laplace"]) for row in result["rows"]))

    def test_multieta_nonpositive_hessian_is_indeterminate(self) -> None:
        module = _load_stress_module()
        model = module.TwoEtaOralPkModel()
        observations = model.predictions("state11", (0.12, -0.08))
        states = tuple(
            pmx.DiscreteState(state_id=state, prior=1.0 / len(module.STATES), label=state)
            for state in module.STATES
        )
        config = pmx.MultiEtaLaplaceConfig(
            omega_covariance=module.OMEGA_COVARIANCE,
            additive_sd=module.ADDITIVE_SD,
            proportional_sd=module.PROPORTIONAL_SD,
        )

        with patch(
            "pmx_discrete_posterior.multieta.finite_difference_hessian",
            return_value=__import__("numpy").diag([-1.0, 1.0]),
        ):
            result = pmx.compute_multieta_posterior(observations, states, model, config)
            self.assertTrue(result.indeterminate)
            self.assertTrue(any("nonpositive_or_nonfinite_hessian" in item for item in result.warnings))
            self.assertTrue(any(diag.indeterminate for diag in result.diagnostics.values()))

    def test_bad_covariance_and_duplicate_states_are_rejected(self) -> None:
        module = _load_stress_module()
        model = module.TwoEtaOralPkModel()
        observations = model.predictions("state11", (0.12, -0.08))
        states = (
            pmx.DiscreteState("state11", 0.5),
            pmx.DiscreteState("state11", 0.5),
        )
        config = pmx.MultiEtaLaplaceConfig(
            omega_covariance=((1.0, 2.0), (2.0, 1.0)),
            additive_sd=module.ADDITIVE_SD,
            proportional_sd=module.PROPORTIONAL_SD,
        )

        with self.assertRaises(ValueError):
            pmx.compute_multieta_posterior(observations, states, model, config)

        good_config = pmx.MultiEtaLaplaceConfig(
            omega_covariance=module.OMEGA_COVARIANCE,
            additive_sd=module.ADDITIVE_SD,
            proportional_sd=module.PROPORTIONAL_SD,
        )
        with self.assertRaises(ValueError):
            pmx.compute_multieta_posterior(observations, states, model, good_config)


if __name__ == "__main__":
    unittest.main()
