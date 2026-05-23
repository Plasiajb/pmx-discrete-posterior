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


STATES = ("cyp3a5_nonexpresser", "cyp3a5_expresser")
STATE_LABELS = {
    "cyp3a5_nonexpresser": "CYP3A5 non-expresser model state",
    "cyp3a5_expresser": "CYP3A5 expresser model state",
}
CL_NONEXPRESSER_L_H = 25.5
CL_EXPRESSER_MULTIPLIER = 1.60
V_L = 113.0
KA_H = 0.35
LAG_H = 0.44
PROPORTIONAL_RUV = 0.183
ETA_CL_CV = 0.295
ETA_V_CV = 0.468
OMEGA_COVARIANCE = (
    (math.log(1.0 + ETA_CL_CV * ETA_CL_CV), 0.0),
    (0.0, math.log(1.0 + ETA_V_CV * ETA_V_CV)),
)
ADDITIVE_SD_NG_ML = 0.05
_PYPLOT = None


def _load_pyplot() -> Any:
    global _PYPLOT
    if _PYPLOT is None:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        _PYPLOT = plt
    return _PYPLOT


@dataclass(frozen=True)
class TacrolimusCyp3a5StressModel:
    """Oral one-compartment steady-state tacrolimus profile model."""

    dose_mg: float = 5.0
    tau_h: float = 12.0
    observation_times_h: tuple[float, ...] = (0.0, 1.0, 2.0, 4.0, 8.0, 12.0)

    def predictions(self, state_id: str, eta: Sequence[float]) -> tuple[float, ...]:
        eta_arr = np.asarray(eta, dtype=float)
        cl = CL_NONEXPRESSER_L_H * self._cl_multiplier(state_id) * math.exp(float(eta_arr[0]))
        volume = V_L * math.exp(float(eta_arr[1]))
        return tuple(
            self._steady_state_oral_ng_ml(float(time_h), cl, volume)
            for time_h in self.observation_times_h
        )

    def _cl_multiplier(self, state_id: str) -> float:
        if state_id == "cyp3a5_nonexpresser":
            return 1.0
        if state_id == "cyp3a5_expresser":
            return CL_EXPRESSER_MULTIPLIER
        raise ValueError(f"Unknown CYP3A5 state: {state_id!r}")

    def _steady_state_oral_ng_ml(self, time_after_dose_h: float, cl_l_h: float, volume_l: float) -> float:
        k_elim = cl_l_h / volume_l
        age_since_absorption_h = (time_after_dose_h - LAG_H) % self.tau_h
        if abs(KA_H - k_elim) < 1e-8:
            numerator = math.exp(-k_elim * age_since_absorption_h)
            repeated_dose_factor = (
                age_since_absorption_h / (1.0 - math.exp(-k_elim * self.tau_h))
                + self.tau_h
                * math.exp(-k_elim * self.tau_h)
                / ((1.0 - math.exp(-k_elim * self.tau_h)) ** 2)
            )
            concentration_mg_l = self.dose_mg * k_elim / volume_l * numerator * repeated_dose_factor
        else:
            concentration_mg_l = self.dose_mg * KA_H / (volume_l * (KA_H - k_elim)) * (
                math.exp(-k_elim * age_since_absorption_h) / (1.0 - math.exp(-k_elim * self.tau_h))
                - math.exp(-KA_H * age_since_absorption_h) / (1.0 - math.exp(-KA_H * self.tau_h))
            )
        return max(float(concentration_mg_l * 1000.0), 0.0)


def build_states() -> tuple[Any, ...]:
    import pmx_discrete_posterior as pmx

    return tuple(
        pmx.DiscreteState(
            state_id=state_id,
            prior=1.0 / len(STATES),
            label=STATE_LABELS[state_id],
            metadata={
                "state_type": "model_conditioned_cyp3a5_phenotype_state",
                "cl_l_h": CL_NONEXPRESSER_L_H
                * (CL_EXPRESSER_MULTIPLIER if state_id == "cyp3a5_expresser" else 1.0),
            },
        )
        for state_id in STATES
    )


def build_config() -> Any:
    import pmx_discrete_posterior as pmx

    return pmx.MultiEtaLaplaceConfig(
        omega_covariance=OMEGA_COVARIANCE,
        additive_sd=ADDITIVE_SD_NG_ML,
        proportional_sd=PROPORTIONAL_RUV,
    )


def build_stress_test(*, gh_nodes: int) -> dict[str, Any]:
    import pmx_discrete_posterior as pmx

    states = build_states()
    config = build_config()
    cases = []
    for true_state in STATES:
        model = TacrolimusCyp3a5StressModel()
        observations = tuple(float(value) for value in model.predictions(true_state, (0.0, 0.0)))
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
                    "case_id": true_state,
                    "true_state": true_state,
                    "state_id": state_id,
                    "state_label": state_result.state.display_label,
                    "prior": state_result.state.prior,
                    "laplace_posterior": state_result.posterior_probability,
                    "quadrature_posterior": quadrature_posterior[state_id],
                    "abs_posterior_diff": abs(state_result.posterior_probability - quadrature_posterior[state_id]),
                    "laplace_log_marginal": diag.laplace_log_likelihood,
                    "quadrature_log_marginal": quadrature_log_marginals[state_id],
                    "eta_hat_cl": diag.eta_hat[0],
                    "eta_hat_v": diag.eta_hat[1],
                    "hessian_logdet": diag.hessian_logdet,
                    "hessian_min_eigenvalue": diag.hessian_min_eigenvalue,
                    "valid_laplace": diag.valid_laplace,
                    "indeterminate": diag.indeterminate,
                    "warnings": ";".join(diag.warnings),
                }
            )
        laplace_map = laplace.map_state_id
        quadrature_map = max(quadrature_posterior, key=quadrature_posterior.get)
        cases.append(
            {
                "case_id": true_state,
                "true_state": true_state,
                "observation_times_h": list(model.observation_times_h),
                "observations_ng_ml": list(observations),
                "laplace_map": laplace_map,
                "quadrature_map": quadrature_map,
                "map_agreement": laplace_map == quadrature_map,
                "max_abs_posterior_diff": max(float(row["abs_posterior_diff"]) for row in rows),
                "laplace_indeterminate": laplace.indeterminate,
                "rows": rows,
            }
        )

    all_rows = [row for case in cases for row in case["rows"]]
    return {
        "model": "model-conditioned tacrolimus CYP3A5 oral one-compartment steady-state stress test",
        "model_parameters": {
            "cl_f_nonexpresser_l_h": CL_NONEXPRESSER_L_H,
            "cl_f_expresser_multiplier": CL_EXPRESSER_MULTIPLIER,
            "v_f_l": V_L,
            "ka_h": KA_H,
            "lag_h": LAG_H,
            "dose_mg": TacrolimusCyp3a5StressModel().dose_mg,
            "tau_h": TacrolimusCyp3a5StressModel().tau_h,
            "proportional_ruv": PROPORTIONAL_RUV,
            "eta_cl_cv": ETA_CL_CV,
            "eta_v_cv": ETA_V_CV,
            "omega_covariance": OMEGA_COVARIANCE,
            "additive_sd_ng_ml_for_numerical_floor": ADDITIVE_SD_NG_ML,
        },
        "literature_anchor": (
            "Bergmann/Hennig/Barraclough/Isbel/Staatz adult kidney-transplant tacrolimus "
            "CYP3A5 PAGE-style parameter summary values supplied for this reproducibility stress test."
        ),
        "boundary_note": (
            "Model-conditioned numerical stress test only. Posterior state labels must not be interpreted "
            "as a clinical CYP3A5 genotype or phenotype test."
        ),
        "gh_nodes_per_dimension": int(gh_nodes),
        "case_count": len(cases),
        "row_count": len(all_rows),
        "map_agreement_all_cases": all(bool(case["map_agreement"]) for case in cases),
        "laplace_indeterminate_any": any(bool(case["laplace_indeterminate"]) for case in cases),
        "max_abs_posterior_diff": max(float(case["max_abs_posterior_diff"]) for case in cases),
        "source": "pmx_discrete_posterior_public_api",
        "cases": cases,
        "rows": all_rows,
    }


def build_trough_curve(*, gh_nodes: int, points: int = 121) -> dict[str, Any]:
    import pmx_discrete_posterior as pmx

    states = build_states()
    config = build_config()
    model = TacrolimusCyp3a5StressModel(observation_times_h=(12.0,))
    state_troughs = {state_id: model.predictions(state_id, (0.0, 0.0))[0] for state_id in STATES}
    lower = max(0.05, min(state_troughs.values()) * 0.35)
    upper = max(state_troughs.values()) * 1.75
    trough_grid = np.linspace(lower, upper, int(points))
    rows = []
    for point_index, trough_ng_ml in enumerate(trough_grid, start=1):
        observations = (float(trough_ng_ml),)
        laplace = pmx.compute_multieta_posterior(observations, states, model, config)
        quadrature_posterior, _ = pmx.gauss_hermite_posterior(
            observations,
            states,
            model,
            config,
            nodes=gh_nodes,
        )
        for state_result in laplace.states:
            state_id = state_result.state.state_id
            rows.append(
                {
                    "point_index": point_index,
                    "trough_ng_ml": float(trough_ng_ml),
                    "state_id": state_id,
                    "laplace_posterior": state_result.posterior_probability,
                    "quadrature_posterior": quadrature_posterior[state_id],
                    "abs_posterior_diff": abs(state_result.posterior_probability - quadrature_posterior[state_id]),
                    "MAP": laplace.map_state_id,
                    "indeterminate": state_result.diagnostics.indeterminate,
                    "warnings": ";".join(state_result.diagnostics.warnings),
                }
            )
    return {
        "x_label": "trough concentration (ng/mL)",
        "state_troughs_ng_ml": state_troughs,
        "rows": rows,
        "points": int(points),
        "gh_nodes_per_dimension": int(gh_nodes),
    }


def write_outputs(
    result: dict[str, Any],
    output_dir: Path,
    *,
    make_plot: bool = True,
    make_trough_curve: bool = True,
    gh_nodes: int = 35,
    trough_gh_nodes: int = 25,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "tacrolimus_cyp3a5_state_posteriors.csv"
    summary_path = output_dir / "tacrolimus_cyp3a5_summary.json"
    png_path = output_dir / "tacrolimus_cyp3a5_laplace_vs_quadrature.png"
    trough_csv_path = output_dir / "tacrolimus_cyp3a5_trough_curve.csv"
    trough_png_path = output_dir / "tacrolimus_cyp3a5_trough_curve.png"

    fieldnames = [
        "case_id",
        "true_state",
        "state_id",
        "state_label",
        "prior",
        "laplace_posterior",
        "quadrature_posterior",
        "abs_posterior_diff",
        "laplace_log_marginal",
        "quadrature_log_marginal",
        "eta_hat_cl",
        "eta_hat_v",
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
        write_state_plot(png_path, result)

    trough_curve: dict[str, Any] | None = None
    if make_trough_curve:
        trough_curve = build_trough_curve(gh_nodes=trough_gh_nodes)
        with trough_csv_path.open("w", newline="", encoding="utf-8") as handle:
            fieldnames_curve = [
                "point_index",
                "trough_ng_ml",
                "state_id",
                "laplace_posterior",
                "quadrature_posterior",
                "abs_posterior_diff",
                "MAP",
                "indeterminate",
                "warnings",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames_curve)
            writer.writeheader()
            writer.writerows(trough_curve["rows"])
        if make_plot:
            write_trough_curve_plot(trough_png_path, trough_curve)

    summary = {key: value for key, value in result.items() if key not in {"rows"}}
    summary["cases"] = [{key: value for key, value in case.items() if key != "rows"} for case in result["cases"]]
    summary["csv"] = str(csv_path)
    summary["png"] = str(png_path) if make_plot else ""
    summary["trough_curve_csv"] = str(trough_csv_path) if make_trough_curve else ""
    summary["trough_curve_png"] = str(trough_png_path) if make_plot and make_trough_curve else ""
    if trough_curve is not None:
        summary["trough_curve"] = {
            "x_label": trough_curve["x_label"],
            "state_troughs_ng_ml": trough_curve["state_troughs_ng_ml"],
            "points": trough_curve["points"],
            "gh_nodes_per_dimension": trough_curve["gh_nodes_per_dimension"],
            "max_abs_posterior_diff": max(float(row["abs_posterior_diff"]) for row in trough_curve["rows"]),
        }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "csv": str(csv_path),
        "summary": str(summary_path),
        "png": str(png_path) if make_plot else "",
        "trough_curve_csv": str(trough_csv_path) if make_trough_curve else "",
        "trough_curve_png": str(trough_png_path) if make_plot and make_trough_curve else "",
    }


def write_state_plot(path: Path, result: dict[str, Any]) -> None:
    plt = _load_pyplot()
    cases = result["cases"]
    fig, axes = plt.subplots(1, len(cases), figsize=(11.2, 4.8), dpi=160, sharey=True)
    if len(cases) == 1:
        axes = [axes]
    width = 0.36
    for ax, case in zip(axes, cases):
        rows = case["rows"]
        x = np.arange(len(rows))
        laplace = [float(row["laplace_posterior"]) for row in rows]
        quadrature = [float(row["quadrature_posterior"]) for row in rows]
        labels = [
            "CYP3A5\nnon-expresser" if row["state_id"] == "cyp3a5_nonexpresser" else "CYP3A5\nexpresser"
            for row in rows
        ]
        ax.bar(x - width / 2, laplace, width, label="Laplace package", color="#2563eb")
        ax.bar(x + width / 2, quadrature, width, label="Gauss-Hermite reference", color="#b45309")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0.0, 1.02)
        generated_state = case["true_state"].replace("cyp3a5_", "").replace("nonexpresser", "non-expresser")
        ax.set_title(f"Generated profile: {generated_state}", fontsize=10)
        ax.grid(axis="y", color="#d0d7de", linewidth=0.8, alpha=0.7)
        ax.text(
            0.03,
            0.88,
            f"max diff={case['max_abs_posterior_diff']:.3g}",
            transform=ax.transAxes,
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "#d0d7de", "alpha": 0.95, "boxstyle": "round,pad=0.25"},
        )
    axes[0].set_ylabel("Posterior probability")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Tacrolimus CYP3A5 model-conditioned stress test, not a clinical genotype test", fontsize=11)
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.94))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def write_trough_curve_plot(path: Path, curve: dict[str, Any]) -> None:
    plt = _load_pyplot()
    rows = curve["rows"]
    fig, ax = plt.subplots(figsize=(8.8, 5.0), dpi=160)
    colors = {"cyp3a5_nonexpresser": "#2563eb", "cyp3a5_expresser": "#b45309"}
    for state_id in STATES:
        state_rows = [row for row in rows if row["state_id"] == state_id]
        ax.plot(
            [float(row["trough_ng_ml"]) for row in state_rows],
            [float(row["laplace_posterior"]) for row in state_rows],
            label=state_id.replace("cyp3a5_", ""),
            linewidth=2.0,
            color=colors[state_id],
        )
    for state_id, trough in curve["state_troughs_ng_ml"].items():
        ax.axvline(float(trough), color=colors[state_id], linestyle="--", linewidth=1.0, alpha=0.65)
        label = "generated expresser trough" if state_id == "cyp3a5_expresser" else "generated non-expresser trough"
        ax.annotate(
            label,
            xy=(float(trough), 0.08 if state_id == "cyp3a5_expresser" else 0.18),
            xytext=(5, 0),
            textcoords="offset points",
            rotation=90,
            va="bottom",
            ha="left",
            fontsize=8,
            color=colors[state_id],
        )
    ax.set_xlabel(curve["x_label"])
    ax.set_ylabel("Laplace posterior probability")
    ax.set_ylim(0.0, 1.02)
    ax.set_title("Trough-only posterior curve: model-conditioned stress test")
    ax.grid(color="#d0d7de", linewidth=0.8, alpha=0.7)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    default_output = (
        WORKSPACE_ROOT
        / "JPKPD_IMA_Laplace_Manuscript_20260521"
        / "supplement"
        / "experiments"
        / "tacrolimus_cyp3a5_stress_test_20260522"
    )
    parser = argparse.ArgumentParser(description="Run a tacrolimus CYP3A5 model-conditioned stress test.")
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument("--gh-nodes", type=int, default=35)
    parser.add_argument("--trough-gh-nodes", type=int, default=25)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--no-trough-curve", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = build_stress_test(gh_nodes=args.gh_nodes)
    outputs = write_outputs(
        result,
        args.output_dir,
        make_plot=not args.no_plot,
        make_trough_curve=not args.no_trough_curve,
        gh_nodes=args.gh_nodes,
        trough_gh_nodes=args.trough_gh_nodes,
    )
    print(json.dumps(outputs, indent=2, sort_keys=True))
    print(
        json.dumps(
            {
                "map_agreement_all_cases": result["map_agreement_all_cases"],
                "max_abs_posterior_diff": result["max_abs_posterior_diff"],
                "laplace_indeterminate_any": result["laplace_indeterminate_any"],
                "boundary_note": result["boundary_note"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
