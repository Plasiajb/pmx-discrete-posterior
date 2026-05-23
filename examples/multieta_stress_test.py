from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


STATES = ("state00", "state01", "state10", "state11")
STATE_LABELS = {
    "state00": "dose 0h missed; dose 12h missed",
    "state01": "dose 0h missed; dose 12h given",
    "state10": "dose 0h given; dose 12h missed",
    "state11": "dose 0h given; dose 12h given",
}
OMEGA_COVARIANCE = ((0.04, 0.012), (0.012, 0.0225))
ADDITIVE_SD = 0.08
PROPORTIONAL_SD = 0.10


@dataclass(frozen=True)
class TwoEtaOralPkModel:
    """Synthetic oral one-compartment model with ETA(CL) and ETA(V)."""

    cl_l_h: float = 4.0
    v_l: float = 50.0
    ka_h: float = 1.2
    dose_mg: float = 300.0
    dose_times_h: tuple[float, ...] = (0.0, 12.0)
    observation_times_h: tuple[float, ...] = (13.5, 16.0, 24.0, 30.0)

    def predictions(self, state_id: str, eta: Sequence[float]) -> tuple[float, ...]:
        eta_arr = np.asarray(eta, dtype=float)
        cl = self.cl_l_h * math.exp(float(eta_arr[0]))
        volume = self.v_l * math.exp(float(eta_arr[1]))
        bits = state_id.removeprefix("state")
        concentrations = []
        for time_h in self.observation_times_h:
            concentration = 0.0
            for bit, dose_time_h in zip(bits, self.dose_times_h):
                if bit == "1":
                    concentration += self._oral_contribution(float(time_h), float(dose_time_h), cl, volume)
            concentrations.append(max(float(concentration), 0.0))
        return tuple(concentrations)

    def _oral_contribution(self, time_h: float, dose_time_h: float, cl_l_h: float, volume_l: float) -> float:
        age_h = float(time_h) - float(dose_time_h)
        if age_h < 0:
            return 0.0
        k_elim = cl_l_h / volume_l
        ka = self.ka_h
        if abs(ka - k_elim) < 1e-8:
            return self.dose_mg * ka / volume_l * age_h * math.exp(-k_elim * age_h)
        return self.dose_mg * ka / (volume_l * (ka - k_elim)) * (
            math.exp(-k_elim * age_h) - math.exp(-ka * age_h)
        )


def build_stress_test(*, gh_nodes: int) -> dict[str, Any]:
    import pmx_discrete_posterior as pmx

    model = TwoEtaOralPkModel()
    true_state = "state11"
    true_eta = (0.12, -0.08)
    observations = tuple(float(value) for value in model.predictions(true_state, true_eta))
    states = tuple(
        pmx.DiscreteState(
            state_id=state,
            prior=1.0 / len(STATES),
            label=STATE_LABELS[state],
            metadata={
                "dose_0h": "given" if state.removeprefix("state")[0] == "1" else "missed",
                "dose_12h": "given" if state.removeprefix("state")[1] == "1" else "missed",
            },
        )
        for state in STATES
    )
    config = pmx.MultiEtaLaplaceConfig(
        omega_covariance=OMEGA_COVARIANCE,
        additive_sd=ADDITIVE_SD,
        proportional_sd=PROPORTIONAL_SD,
    )
    laplace = pmx.compute_multieta_posterior(observations, states, model, config)
    quadrature_posterior, quadrature_log_marginals = pmx.gauss_hermite_posterior(
        observations,
        states,
        model,
        config,
        nodes=gh_nodes,
    )
    rows = []
    for state_result in laplace.states:
        state_id = state_result.state.state_id
        diag = state_result.diagnostics
        rows.append(
            {
                "state_id": state_id,
                "state_label": state_result.state.display_label,
                "prior": state_result.state.prior,
                "laplace_posterior": state_result.posterior_probability,
                "quadrature_posterior": quadrature_posterior[state_id],
                "abs_posterior_diff": abs(state_result.posterior_probability - quadrature_posterior[state_id]),
                "laplace_log_marginal": diag.laplace_log_likelihood,
                "quadrature_log_marginal": quadrature_log_marginals[state_id],
                "eta_hat_1_cl": diag.eta_hat[0],
                "eta_hat_2_v": diag.eta_hat[1],
                "hessian_logdet": diag.hessian_logdet,
                "hessian_min_eigenvalue": diag.hessian_min_eigenvalue,
                "valid_laplace": diag.valid_laplace,
                "indeterminate": diag.indeterminate,
                "warnings": ";".join(diag.warnings),
            }
        )
    laplace_map = laplace.map_state_id
    quadrature_map = max(quadrature_posterior, key=quadrature_posterior.get)
    max_abs_diff = max(float(row["abs_posterior_diff"]) for row in rows)
    return {
        "model": "synthetic oral one-compartment PK with ETA(CL), ETA(V), four observations, combined RUV",
        "true_state": true_state,
        "true_eta": list(true_eta),
        "observation_times_h": list(model.observation_times_h),
        "observations": list(observations),
        "omega_covariance": OMEGA_COVARIANCE,
        "additive_sd": ADDITIVE_SD,
        "proportional_sd": PROPORTIONAL_SD,
        "laplace_map": laplace_map,
        "quadrature_map": quadrature_map,
        "map_agreement": laplace_map == quadrature_map,
        "max_abs_posterior_diff": max_abs_diff,
        "gh_nodes_per_dimension": int(gh_nodes),
        "laplace_indeterminate": laplace.indeterminate,
        "rows": rows,
        "source": "pmx_discrete_posterior_public_api",
        "boundary_note": (
            "Supplement stress test only. It checks multi-ETA/multi-observation/combined-RUV "
            "Laplace behavior against numerical quadrature in a synthetic low-dimensional model."
        ),
    }


def write_outputs(result: dict[str, Any], output_dir: Path, *, make_plot: bool = True) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "multieta_stress_test_state_posteriors.csv"
    summary_path = output_dir / "multieta_stress_test_summary.json"
    png_path = output_dir / "multieta_stress_test_laplace_vs_quadrature.png"
    fieldnames = [
        "state_id",
        "state_label",
        "prior",
        "laplace_posterior",
        "quadrature_posterior",
        "abs_posterior_diff",
        "laplace_log_marginal",
        "quadrature_log_marginal",
        "eta_hat_1_cl",
        "eta_hat_2_v",
        "hessian_logdet",
        "hessian_min_eigenvalue",
        "valid_laplace",
        "indeterminate",
        "warnings",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result["rows"])
    if make_plot:
        write_plot(png_path, result)
    summary = {key: value for key, value in result.items() if key != "rows"}
    summary["csv"] = str(csv_path)
    summary["png"] = str(png_path) if make_plot else ""
    summary["row_count"] = len(result["rows"])
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return {"csv": str(csv_path), "summary": str(summary_path), "png": str(png_path) if make_plot else ""}


def write_plot(path: Path, result: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = result["rows"]
    x = np.arange(len(rows))
    width = 0.36
    laplace = [float(row["laplace_posterior"]) for row in rows]
    quadrature = [float(row["quadrature_posterior"]) for row in rows]
    labels = [str(row["state_id"]) for row in rows]
    fig, ax = plt.subplots(figsize=(9.2, 5.4), dpi=160)
    ax.bar(x - width / 2, laplace, width, label="Laplace package", color="#1f6feb")
    ax.bar(x + width / 2, quadrature, width, label="Gauss-Hermite reference", color="#c47f00")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel("Posterior probability")
    ax.set_title("Supplement stress test: 2-ETA, 4-observation, combined RUV")
    ax.text(
        0.012,
        0.94,
        f"max abs diff={result['max_abs_posterior_diff']:.3g}; MAP agreement={result['map_agreement']}",
        transform=ax.transAxes,
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#d0d7de", "alpha": 0.95, "boxstyle": "round,pad=0.35"},
    )
    ax.grid(axis="y", color="#d0d7de", linewidth=0.8, alpha=0.7)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    default_output = (
        WORKSPACE_ROOT
        / "JPKPD_IMA_Laplace_Manuscript_20260521"
        / "supplement"
        / "experiments"
        / "multieta_stress_test_20260521"
    )
    parser = argparse.ArgumentParser(description="Run a multi-ETA Laplace supplement stress test.")
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument("--gh-nodes", type=int, default=35)
    parser.add_argument("--no-plot", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = build_stress_test(gh_nodes=args.gh_nodes)
    outputs = write_outputs(result, args.output_dir, make_plot=not args.no_plot)
    print(json.dumps(outputs, indent=2, sort_keys=True))
    print(
        json.dumps(
            {
                "map_agreement": result["map_agreement"],
                "max_abs_posterior_diff": result["max_abs_posterior_diff"],
                "laplace_indeterminate": result["laplace_indeterminate"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

