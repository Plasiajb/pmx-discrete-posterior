from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
SRC_ROOT = PROJECT_ROOT / "src"
EXAMPLE_PATH = PROJECT_ROOT / "examples" / "tacrolimus_staatz_2comp_tier1_stress_test.py"
TEST_OUTPUT_DIR = (
    PROJECT_ROOT
    / "tests"
    / "_tmp_tacrolimus_staatz_2comp_tier1_api"
)
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _load_stress_module():
    spec = importlib.util.spec_from_file_location("tacrolimus_staatz_2comp_tier1_stress_test", EXAMPLE_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Could not load example module from {EXAMPLE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TacrolimusStaatzTwoCompTier1ApiTests(unittest.TestCase):
    def test_profile_laplace_matches_gauss_hermite_reference(self) -> None:
        module = _load_stress_module()
        result = module.build_stress_test(gh_nodes=41)

        self.assertEqual(result["case_count"], 2)
        self.assertEqual(result["profile_row_count"], 12)
        self.assertTrue(result["map_agreement_all_cases"])
        self.assertFalse(result["laplace_indeterminate_any"])
        self.assertLessEqual(result["max_laplace_normalization_error"], 1e-8)
        self.assertLessEqual(result["max_quadrature_normalization_error"], 1e-8)
        self.assertLessEqual(result["max_abs_posterior_diff"], 0.02)
        self.assertIn("must not be interpreted as a clinical CYP3A5 genotype", result["boundary_note"])
        self.assertIn("depot -> central -> peripheral", result["model_structure"])

        for case in result["cases"]:
            self.assertEqual(case["laplace_map"], case["true_state"])
            self.assertEqual(case["quadrature_map"], case["true_state"])
            self.assertFalse(case["laplace_indeterminate"])
            self.assertAlmostEqual(sum(float(row["laplace_posterior"]) for row in case["rows"]), 1.0, places=10)
            self.assertAlmostEqual(sum(float(row["quadrature_posterior"]) for row in case["rows"]), 1.0, places=10)
            self.assertTrue(all(bool(row["valid_laplace"]) for row in case["rows"]))
            self.assertTrue(all(not bool(row["indeterminate"]) for row in case["rows"]))

    def test_deterministic_profiles_are_two_state_six_time_outputs(self) -> None:
        module = _load_stress_module()
        rows = module.build_profile_rows()

        self.assertEqual(len(rows), 12)
        self.assertEqual({row["state_id"] for row in rows}, set(module.STATES))
        self.assertEqual(
            [row["time_after_dose_h"] for row in rows if row["state_id"] == "cyp3a5_nonexpresser"],
            list(module.OBSERVATION_TIMES_H),
        )
        self.assertTrue(all(float(row["concentration_ng_ml"]) > 0.0 for row in rows))
        troughs = {
            row["state_id"]: float(row["concentration_ng_ml"])
            for row in rows
            if float(row["time_after_dose_h"]) == 12.0
        }
        self.assertGreater(troughs["cyp3a5_nonexpresser"], troughs["cyp3a5_expresser"])

    def test_trough_curve_normalizes_and_has_no_indeterminate_rows(self) -> None:
        module = _load_stress_module()
        curve = module.build_trough_curve(gh_nodes=25, points=21)

        by_point: dict[int, list[dict[str, object]]] = {}
        for row in curve["rows"]:
            by_point.setdefault(int(row["point_index"]), []).append(row)

        self.assertEqual(len(by_point), 21)
        self.assertEqual(set(curve["state_troughs_ng_ml"]), set(module.STATES))
        self.assertFalse(curve["indeterminate_any"])
        for rows in by_point.values():
            self.assertEqual(len(rows), 2)
            self.assertAlmostEqual(sum(float(row["laplace_posterior"]) for row in rows), 1.0, places=10)
            self.assertAlmostEqual(sum(float(row["quadrature_posterior"]) for row in rows), 1.0, places=10)
            self.assertTrue(all(not bool(row["indeterminate"]) for row in rows))

    def test_writes_expected_outputs_with_plots(self) -> None:
        module = _load_stress_module()
        result = module.build_stress_test(gh_nodes=25)

        outputs = module.write_outputs(
            result,
            TEST_OUTPUT_DIR,
            make_plot=True,
            make_trough_curve=True,
            gh_nodes=25,
            trough_gh_nodes=15,
        )

        self.assertTrue(Path(outputs["csv"]).is_file())
        self.assertTrue(Path(outputs["profile_csv"]).is_file())
        self.assertTrue(Path(outputs["summary"]).is_file())
        self.assertTrue(Path(outputs["posterior_png"]).is_file())
        self.assertTrue(Path(outputs["profile_png"]).is_file())
        self.assertTrue(Path(outputs["trough_curve_csv"]).is_file())
        self.assertTrue(Path(outputs["trough_curve_png"]).is_file())


if __name__ == "__main__":
    unittest.main()
