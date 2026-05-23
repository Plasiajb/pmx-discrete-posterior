from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
SRC_ROOT = PROJECT_ROOT / "src"
EXAMPLE_PATH = PROJECT_ROOT / "examples" / "tacrolimus_cyp3a5_stress_test.py"
TEST_OUTPUT_DIR = (
    PROJECT_ROOT
    / "tests"
    / "_tmp_tacrolimus_cyp3a5_stress_api"
)
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _load_stress_module():
    spec = importlib.util.spec_from_file_location("tacrolimus_cyp3a5_stress_test", EXAMPLE_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Could not load example module from {EXAMPLE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TacrolimusCyp3a5StressApiTests(unittest.TestCase):
    def test_tacrolimus_cyp3a5_laplace_matches_gauss_hermite_reference(self) -> None:
        module = _load_stress_module()
        result = module.build_stress_test(gh_nodes=35)

        self.assertEqual(result["case_count"], 2)
        self.assertTrue(result["map_agreement_all_cases"])
        self.assertFalse(result["laplace_indeterminate_any"])
        self.assertLess(result["max_abs_posterior_diff"], 0.01)
        self.assertIn("not be interpreted as a clinical CYP3A5 genotype", result["boundary_note"])

        for case in result["cases"]:
            self.assertEqual(case["laplace_map"], case["true_state"])
            self.assertEqual(case["quadrature_map"], case["true_state"])
            self.assertFalse(case["laplace_indeterminate"])
            self.assertAlmostEqual(sum(float(row["laplace_posterior"]) for row in case["rows"]), 1.0, places=10)
            self.assertAlmostEqual(sum(float(row["quadrature_posterior"]) for row in case["rows"]), 1.0, places=10)
            self.assertTrue(all(bool(row["valid_laplace"]) for row in case["rows"]))
            self.assertTrue(all(not bool(row["indeterminate"]) for row in case["rows"]))

    def test_tacrolimus_cyp3a5_trough_curve_normalizes(self) -> None:
        module = _load_stress_module()
        curve = module.build_trough_curve(gh_nodes=25, points=21)

        by_point: dict[int, list[dict[str, object]]] = {}
        for row in curve["rows"]:
            by_point.setdefault(int(row["point_index"]), []).append(row)

        self.assertEqual(len(by_point), 21)
        self.assertEqual(set(curve["state_troughs_ng_ml"]), set(module.STATES))
        for rows in by_point.values():
            self.assertEqual(len(rows), 2)
            self.assertAlmostEqual(sum(float(row["laplace_posterior"]) for row in rows), 1.0, places=10)
            self.assertAlmostEqual(sum(float(row["quadrature_posterior"]) for row in rows), 1.0, places=10)
            self.assertTrue(all(not bool(row["indeterminate"]) for row in rows))

    def test_tacrolimus_cyp3a5_writes_expected_outputs_with_plots(self) -> None:
        module = _load_stress_module()
        result = module.build_stress_test(gh_nodes=25)

        outputs = module.write_outputs(
            result,
            TEST_OUTPUT_DIR,
            make_plot=True,
            make_trough_curve=True,
            gh_nodes=15,
            trough_gh_nodes=15,
        )

        self.assertTrue(Path(outputs["csv"]).is_file())
        self.assertTrue(Path(outputs["summary"]).is_file())
        self.assertTrue(Path(outputs["png"]).is_file())
        self.assertTrue(Path(outputs["trough_curve_csv"]).is_file())
        self.assertTrue(Path(outputs["trough_curve_png"]).is_file())


if __name__ == "__main__":
    unittest.main()
