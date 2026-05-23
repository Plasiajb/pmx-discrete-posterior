from __future__ import annotations

import csv
import importlib.util
import shutil
import sys
import unittest
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_PATH = PROJECT_ROOT / "examples" / "oxc4_pmix_dv_plot.py"
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _load_example_module():
    spec = importlib.util.spec_from_file_location("oxc4_pmix_dv_plot", EXAMPLE_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Could not load example module from {EXAMPLE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Oxc4PmixDvExampleTests(unittest.TestCase):
    def test_example_generates_normalized_curve_csv_summary_and_png(self) -> None:
        module = _load_example_module()
        output_dir = PROJECT_ROOT / "tests" / "_tmp_oxc4_pmix_dv_example"
        if output_dir.exists():
            shutil.rmtree(output_dir)
        try:
            code = module.main(["--output-dir", str(output_dir), "--points", "9"])
            self.assertEqual(code, 0)

            csv_path = output_dir / "oxc4_pmix_dv_curve.csv"
            png_path = output_dir / "oxc4_pmix_dv_curve.png"
            summary_path = output_dir / "summary.json"
            self.assertTrue(csv_path.is_file())
            self.assertTrue(png_path.is_file())
            self.assertGreater(png_path.stat().st_size, 0)
            self.assertTrue(summary_path.is_file())

            with csv_path.open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertGreaterEqual(len(rows), 9 * 4)
            by_point: dict[str, list[dict[str, str]]] = defaultdict(list)
            for row in rows:
                by_point[row["point_index"]].append(row)
                for key in ("DV", "scenario_key", "PMIX", "eta_hat", "hessian", "laplace_log_likelihood", "MAP"):
                    self.assertIn(key, row)

            for point_rows in by_point.values():
                self.assertEqual(len(point_rows), 4)
                self.assertAlmostEqual(sum(float(row["PMIX"]) for row in point_rows), 1.0, places=10)
        finally:
            if output_dir.exists():
                shutil.rmtree(output_dir)


if __name__ == "__main__":
    unittest.main()
