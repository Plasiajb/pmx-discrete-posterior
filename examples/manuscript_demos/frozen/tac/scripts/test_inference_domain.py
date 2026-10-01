"""Entry-level covariance rejection with core spies and analytic SPD controls."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import unittest
from unittest.mock import patch

import numpy as np

from inference_smoke import bootstrap
from supported_inference import inference_entry

REJECTIONS = []


class SupportedEntryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pmx = bootstrap(Path(__file__).resolve().parents[1])
        import routes
        cls.routes = routes

    def rejected_before_core(self, omega, dimension):
        cfg = self.pmx.MultiEtaLaplaceConfig(omega_covariance=omega,additive_sd=0.5)
        case = SimpleNamespace(error_config=SimpleNamespace(public_config=lambda:cfg))
        for route in ("laplace","adaptive_ghq","tacrolimus_laplace","tacrolimus_prior_ghq"):
            with self.subTest(route=route), \
                 patch.object(self.pmx,"compute_multieta_posterior") as core, \
                 patch.object(self.pmx,"multieta_state_diagnostics") as mode, \
                 patch.object(self.routes,"laplace_route") as tac_laplace, \
                 patch.object(self.routes,"prior_ghq") as tac_ghq:
                with self.assertRaises(ValueError) as rejected:
                    inference_entry(route,config=cfg,eta_dimension=dimension,observations=[0.],
                        states=(),model=None,case=case,nodes=41)
                core.assert_not_called()
                mode.assert_not_called()
                tac_laplace.assert_not_called()
                tac_ghq.assert_not_called()
                REJECTIONS.append({"fixture":self.id().split(".")[-1],"route":route,
                    "expected_eta_dimension":dimension,"outcome":"rejected",
                    "error":str(rejected.exception),"frozen_core_calls":core.call_count,
                    "frozen_mode_calls":mode.call_count,"study_laplace_calls":tac_laplace.call_count,
                    "study_ghq_calls":tac_ghq.call_count})

    def test_negative_definite_2d_entry(self):
        self.rejected_before_core(-np.eye(2),2)

    def test_indefinite_positive_determinant_3d_entry(self):
        omega = np.diag([-1.,-2.,3.])
        self.assertGreater(np.linalg.det(omega),0)
        self.rejected_before_core(omega,3)

    def test_nonsymmetric_entry(self):
        self.rejected_before_core([[1,0.4],[0.1,1]],2)

    def test_singular_entry(self):
        self.rejected_before_core([[1,1],[1,1]],2)

    def test_nonfinite_entry(self):
        for value in (np.nan,np.inf,-np.inf):
            self.rejected_before_core([[1,0],[0,value]],2)

    def test_dimension_mismatch_entry(self):
        self.rejected_before_core(np.eye(3),2)
        self.rejected_before_core([[1.,0.]],2)
        self.rejected_before_core([],2)

    def analytic(self, omega):
        y,sd = np.array([0.7,-0.2]),0.5
        cfg = self.pmx.MultiEtaLaplaceConfig(omega_covariance=tuple(map(tuple,omega)),
            additive_sd=sd,proportional_sd=0.,optimizer_method="L-BFGS-B")

        class LinearGaussian:
            def predictions(self,state_id,eta):
                return tuple(eta)

        state = self.pmx.DiscreteState(state_id="one",prior=1.)
        with patch.object(self.pmx,"compute_multieta_posterior",wraps=self.pmx.compute_multieta_posterior) as core:
            result = inference_entry("laplace",config=cfg,eta_dimension=2,
                observations=y,states=(state,),model=LinearGaussian())
            core.assert_called_once()
        total = omega + sd**2*np.eye(2)
        analytic = -np.log(2*np.pi)-0.5*np.linalg.slogdet(total)[1]-0.5*y @ np.linalg.solve(total,y)
        self.assertAlmostEqual(result.log_evidence,float(analytic),delta=1e-6)
        self.assertEqual(result.posterior,{"one":1.})

    def test_valid_diagonal_spd_analytic_entry(self):
        self.analytic(np.diag([0.4,0.25]))

    def test_valid_correlated_spd_analytic_entry(self):
        self.analytic(np.array([[0.4,0.1],[0.1,0.25]]))


if __name__ == "__main__":
    program = unittest.main(verbosity=2,exit=False)
    root = Path(__file__).resolve().parents[1]
    files = [root/"scripts/supported_inference.py",root/"scripts/inference_smoke.py",
             root/"scripts/precision_reproduce.py"]
    files += list((root/"source/pmx_discrete_posterior_v0.1.4/src/pmx_discrete_posterior").glob("*.py"))
    print(json.dumps({"status":"PASS" if program.result.wasSuccessful() else "FAIL",
        "supported_entry":"supported_inference.inference_entry",
        "rejected_calls":REJECTIONS,"negative_entry_call_count":len(REJECTIONS),
        "positive_controls":["diagonal SPD analytic Gaussian","correlated SPD analytic Gaussian"],
        "source_sha256":{p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        "generic_release_bug_fixed":False,"raw_package_API_protected":False},indent=2))
    raise SystemExit(0 if program.result.wasSuccessful() else 1)
