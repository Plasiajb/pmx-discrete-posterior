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

THETA_CL_F_L_H = 25.5
THETA_CYP3A5 = 1.60
THETA_V1_F_L = 113.0
Q_F_L_H = 67.9
V2_F_L = 1060.0
KA_H = 0.35
ALAG_H = 0.44
WT_KG = 70.0
HAEM = 0.33
POD_DAYS = 22.7
PREDCMAX_FREE_NMOL_L = 155.5
DOSE_MG = 5.0
TAU_H = 12.0
OBSERVATION_TIMES_H = (0.0, 1.0, 2.0, 4.0, 8.0, 12.0)
STEADY_STATE_PRIOR_INTERVALS = 120

PROPORTIONAL_RUV = 0.183
ADDITIVE_SD_NG_ML = 0.05
ETA_CL_CV = 0.295
ETA_V1_CV = 0.468
OMEGA_COVARIANCE = (
    (math.log(1.0 + ETA_CL_CV * ETA_CL_CV), 0.0),
    (0.0, math.log(1.0 + ETA_V1_CV * ETA_V1_CV)),
)

_PYPLOT = None


def _load_pyplot() -> Any:
    global _PYPLOT
    if _PYPLOT is None:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        _PYPLOT = plt
    return _PYPLOT


def _matrix_exponential(matrix: np.ndarray, time_h: float) -> np.ndarray:
    try:
        from scipy.linalg import expm

        return np.asarray(expm(matrix * float(time_h)), dtype=float)
    except Exception:
        eigvals, eigvecs = np.linalg.eig(matrix)
        inv_eigvecs = np.linalg.inv(eigvecs)
        exp_diag = np.diag(np.exp(eigvals * float(time_h)))
        return np.real_if_close(eigvecs @ exp_diag @ inv_eigvecs).astype(float)


@dataclass(frozen=True)
class TacrolimusStaatzTwoCompTier1Model:
    """Strict structural two-compartment oral tacrolimus model for Tier 1 checks."""

    dose_mg: float = DOSE_MG
    tau_h: float = TAU_H
    observation_times_h: tuple[float, ...] = OBSERVATION_TIMES_H
    steady_state_prior_intervals: int = STEADY_STATE_PRIOR_INTERVALS

    def predictions(self, state_id: str, eta: Sequence[float]) -> tuple[float, ...]:
        eta_arr = np.asarray(eta, dtype=float)
        if eta_arr.size != 2:
            raise ValueError("Tacrolimus Tier 1 model requires ETA(CL) and ETA(V1).")
        cl_f = self.typical_cl_f_l_h(state_id) * math.exp(float(eta_arr[0]))
        v1_f = THETA_V1_F_L * math.exp(float(eta_arr[1]))
        return tuple(
            self._repeated_dose_concentration_ng_ml(float(time_h), cl_f, v1_f)
            for time_h in self.observation_times_h
        )

    @staticmethod
    def cyp3a5_x(state_id: str) -> int:
        if state_id == "cyp3a5_nonexpresser":
            return 0
        if state_id == "cyp3a5_expresser":
            return 1
        raise ValueError(f"Unknown CYP3A5 state: {state_id!r}")

    def typical_cl_f_l_h(self, state_id: str) -> float:
        # Covariates are fixed at centering values, so only CYP3A5 changes CL/F.
        return THETA_CL_F_L_H * (THETA_CYP3A5 ** self.cyp3a5_x(state_id))

    def _system_matrix(self, cl_f_l_h: float, v1_f_l: float) -> np.ndarray:
        return np.asarray(
            [
                [-KA_H, 0.0, 0.0],
                [KA_H, -((cl_f_l_h + Q_F_L_H) / v1_f_l), Q_F_L_H / V2_F_L],
                [0.0, Q_F_L_H / v1_f_l, -(Q_F_L_H / V2_F_L)],
            ],
            dtype=float,
        )

    def _repeated_dose_concentration_ng_ml(self, time_after_dose_h: float, cl_f_l_h: float, v1_f_l: float) -> float:
        matrix = self._system_matrix(cl_f_l_h, v1_f_l)
        initial = np.asarray([self.dose_mg, 0.0, 0.0], dtype=float)
        first_elapsed_h = float(time_after_dose_h) - ALAG_H
        start_index = max(0, int(math.ceil((-first_elapsed_h) / self.tau_h))) if first_elapsed_h < 0.0 else 0
        count = int(self.steady_state_prior_intervals) - start_index + 1
        if count <= 0:
            return 0.0

        base_elapsed_h = first_elapsed_h + float(start_index) * self.tau_h
        base_transition = _matrix_exponential(matrix, base_elapsed_h)
        tau_transition = _matrix_exponential(matrix, self.tau_h)
        identity = np.eye(3, dtype=float)
        transition_power = np.linalg.matrix_power(tau_transition, count)
        summed_initial = np.linalg.solve(identity - tau_transition, (identity - transition_power) @ initial)
        amount = base_transition @ summed_initial
        central_mg = float(amount[1])
        return max(float((central_mg / v1_f_l) * 1000.0), 0.0)


def build_states() -> tuple[Any, ...]:
    import pmx_discrete_posterior as pmx

    model = TacrolimusStaatzTwoCompTier1Model()
    return tuple(
        pmx.DiscreteState(
            state_id=state_id,
            prior=1.0 / len(STATES),
            label=STATE_LABELS[state_id],
            metadata={
                "state_type": "model_conditioned_cyp3a5_phenotype_state",
                "cyp3a5_x": model.cyp3a5_x(state_id),
                "typical_cl_f_l_h": model.typical_cl_f_l_h(state_id),
                "typical_v1_f_l": THETA_V1_F_L,
                "q_f_l_h": Q_F_L_H,
                "v2_f_l": V2_F_L,
                "ka_h": KA_H,
                "alag_h": ALAG_H,
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


def build_profile_rows(model: TacrolimusStaatzTwoCompTier1Model | None = None) -> list[dict[str, Any]]:
    model = model or TacrolimusStaatzTwoCompTier1Model()
    rows: list[dict[str, Any]] = []
    for state_id in STATES:
        predictions = model.predictions(state_id, (0.0, 0.0))
        for time_h, concentration in zip(model.observation_times_h, predictions):
            rows.append(
                {
                    "state_id": state_id,
                    "state_label": STATE_LABELS[state_id],
                    "time_after_dose_h": float(time_h),
                    "concentration_ng_ml": float(concentration),
                    "dose_mg": model.dose_mg,
                    "tau_h": model.tau_h,
                    "steady_state_prior_intervals": model.steady_state_prior_intervals,
                    "eta_cl": 0.0,
                    "eta_v1": 0.0,
                }
            )
    return rows


def _case_result(
    *,
    true_state: str,
    model: TacrolimusStaatzTwoCompTier1Model,
    states: Sequence[Any],
    config: Any,
    gh_nodes: int,
) -> dict[str, Any]:
    import pmx_discrete_posterior as pmx

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
                "eta_hat_v1": diag.eta_hat[1],
                "hessian_logdet": diag.hessian_logdet,
                "hessian_min_eigenvalue": diag.hessian_min_eigenvalue,
                "valid_laplace": diag.valid_laplace,
                "indeterminate": diag.indeterminate,
                "warnings": ";".join(diag.warnings),
            }
        )
    laplace_sum = sum(float(row["laplace_posterior"]) for row in rows)
    quadrature_sum = sum(float(row["quadrature_posterior"]) for row in rows)
    laplace_map = laplace.map_state_id
    quadrature_map = max(quadrature_posterior, key=quadrature_posterior.get)
    return {
        "case_id": true_state,
        "true_state": true_state,
        "observation_times_h": list(model.observation_times_h),
        "observations_ng_ml": list(observations),
        "laplace_map": laplace_map,
        "quadrature_map": quadrature_map,
        "map_agreement": laplace_map == quadrature_map,
        "max_abs_posterior_diff": max(float(row["abs_posterior_diff"]) for row in rows),
        "laplace_normalization_error": abs(laplace_sum - 1.0),
        "quadrature_normalization_error": abs(quadrature_sum - 1.0),
        "laplace_indeterminate": laplace.indeterminate,
        "rows": rows,
    }


def build_stress_test(*, gh_nodes: int) -> dict[str, Any]:
    states = build_states()
    config = build_config()
    model = TacrolimusStaatzTwoCompTier1Model()
    cases = [
        _case_result(true_state=true_state, model=model, states=states, config=config, gh_nodes=gh_nodes)
        for true_state in STATES
    ]
    all_rows = [row for case in cases for row in case["rows"]]
    profile_rows = build_profile_rows(model)
    return {
        "model": "Staatz-priority tacrolimus Tier 1 strict structural two-compartment oral model",
        "model_structure": "depot -> central -> peripheral with absorption lag, first-order absorption, first-order elimination, and first-order intercompartmental distribution",
        "model_parameters": {
            "theta_cl_f_l_h": THETA_CL_F_L_H,
            "theta_cyp3a5_cl_multiplier": THETA_CYP3A5,
            "theta_v1_f_l": THETA_V1_F_L,
            "q_f_l_h": Q_F_L_H,
            "v2_f_l": V2_F_L,
            "ka_h": KA_H,
            "alag_h": ALAG_H,
            "dose_mg": model.dose_mg,
            "tau_h": model.tau_h,
            "observation_times_h": list(model.observation_times_h),
            "steady_state_prior_intervals": model.steady_state_prior_intervals,
            "proportional_ruv": PROPORTIONAL_RUV,
            "additive_sd_ng_ml": ADDITIVE_SD_NG_ML,
            "eta_cl_cv": ETA_CL_CV,
            "eta_v1_cv": ETA_V1_CV,
            "omega_covariance": OMEGA_COVARIANCE,
        },
        "fixed_covariates": {
            "WT_kg": WT_KG,
            "HAEM": HAEM,
            "POD_days": POD_DAYS,
            "PredCmax_free_nmol_l": PREDCMAX_FREE_NMOL_L,
        },
        "steady_state_approximation": (
            "Linear repeated-dose simulation sums the last "
            f"{model.steady_state_prior_intervals} q12h prior intervals plus the index dose, "
            "with each dose entering the absorption depot after ALAG."
        ),
        "literature_anchor": (
            "Bergmann, Hennig, Barraclough, Isbel, and Staatz adult kidney-transplant tacrolimus "
            "CYP3A5 model structure and public PAGE-style parameter summary values."
        ),
        "boundary_note": (
            "Reduced stochastic Tier 1 numerical stress test only. It estimates model-conditioned "
            "CYP3A5 state probabilities and must not be interpreted as a clinical CYP3A5 genotype, "
            "phenotype, dose recommendation, or NONMEM PMIX reference."
        ),
        "gh_nodes_per_dimension": int(gh_nodes),
        "case_count": len(cases),
        "row_count": len(all_rows),
        "profile_row_count": len(profile_rows),
        "map_agreement_all_cases": all(bool(case["map_agreement"]) for case in cases),
        "laplace_indeterminate_any": any(bool(case["laplace_indeterminate"]) for case in cases),
        "max_abs_posterior_diff": max(float(case["max_abs_posterior_diff"]) for case in cases),
        "max_laplace_normalization_error": max(float(case["laplace_normalization_error"]) for case in cases),
        "max_quadrature_normalization_error": max(float(case["quadrature_normalization_error"]) for case in cases),
        "source": "pmx_discrete_posterior_public_api",
        "cases": cases,
        "rows": all_rows,
        "profile_rows": profile_rows,
    }


def build_trough_curve(*, gh_nodes: int, points: int = 121) -> dict[str, Any]:
    import pmx_discrete_posterior as pmx

    states = build_states()
    config = build_config()
    model = TacrolimusStaatzTwoCompTier1Model(observation_times_h=(12.0,))
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
        "x_label": "trough concentration at 12 h after dose (ng/mL)",
        "state_troughs_ng_ml": state_troughs,
        "rows": rows,
        "points": int(points),
        "gh_nodes_per_dimension": int(gh_nodes),
        "max_abs_posterior_diff": max(float(row["abs_posterior_diff"]) for row in rows),
        "indeterminate_any": any(bool(row["indeterminate"]) for row in rows),
    }


def write_outputs(
    result: dict[str, Any],
    output_dir: Path,
    *,
    make_plot: bool = True,
    make_trough_curve: bool = True,
    gh_nodes: int = 41,
    trough_gh_nodes: int = 25,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "tacrolimus_staatz_2comp_tier1_state_posteriors.csv"
    profile_csv_path = output_dir / "tacrolimus_staatz_2comp_tier1_deterministic_profiles.csv"
    summary_path = output_dir / "tacrolimus_staatz_2comp_tier1_summary.json"
    posterior_png_path = output_dir / "tacrolimus_staatz_2comp_tier1_laplace_vs_quadrature.png"
    profile_png_path = output_dir / "tacrolimus_staatz_2comp_tier1_profiles.png"
    trough_csv_path = output_dir / "tacrolimus_staatz_2comp_tier1_trough_curve.csv"
    trough_png_path = output_dir / "tacrolimus_staatz_2comp_tier1_trough_curve.png"

    posterior_fields = [
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
        "eta_hat_v1",
        "hessian_logdet",
        "hessian_min_eigenvalue",
        "valid_laplace",
        "indeterminate",
        "warnings",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=posterior_fields)
        writer.writeheader()
        writer.writerows(result["rows"])

    profile_fields = [
        "state_id",
        "state_label",
        "time_after_dose_h",
        "concentration_ng_ml",
        "dose_mg",
        "tau_h",
        "steady_state_prior_intervals",
        "eta_cl",
        "eta_v1",
    ]
    with profile_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=profile_fields)
        writer.writeheader()
        writer.writerows(result["profile_rows"])

    if make_plot:
        write_state_plot(posterior_png_path, result)
        write_profile_plot(profile_png_path, result)

    trough_curve: dict[str, Any] | None = None
    if make_trough_curve:
        trough_curve = build_trough_curve(gh_nodes=trough_gh_nodes)
        trough_fields = [
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
        with trough_csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=trough_fields)
            writer.writeheader()
            writer.writerows(trough_curve["rows"])
        if make_plot:
            write_trough_curve_plot(trough_png_path, trough_curve)

    summary = {key: value for key, value in result.items() if key not in {"rows", "profile_rows"}}
    summary["cases"] = [{key: value for key, value in case.items() if key != "rows"} for case in result["cases"]]
    summary["csv"] = str(csv_path)
    summary["profile_csv"] = str(profile_csv_path)
    summary["posterior_png"] = str(posterior_png_path) if make_plot else ""
    summary["profile_png"] = str(profile_png_path) if make_plot else ""
    summary["trough_curve_csv"] = str(trough_csv_path) if make_trough_curve else ""
    summary["trough_curve_png"] = str(trough_png_path) if make_plot and make_trough_curve else ""
    if trough_curve is not None:
        summary["trough_curve"] = {
            "x_label": trough_curve["x_label"],
            "state_troughs_ng_ml": trough_curve["state_troughs_ng_ml"],
            "points": trough_curve["points"],
            "gh_nodes_per_dimension": trough_curve["gh_nodes_per_dimension"],
            "max_abs_posterior_diff": trough_curve["max_abs_posterior_diff"],
            "indeterminate_any": trough_curve["indeterminate_any"],
        }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "csv": str(csv_path),
        "profile_csv": str(profile_csv_path),
        "summary": str(summary_path),
        "posterior_png": str(posterior_png_path) if make_plot else "",
        "profile_png": str(profile_png_path) if make_plot else "",
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
    colors = {"laplace": "#2563eb", "quadrature": "#b45309"}
    for ax, case in zip(axes, cases):
        rows = case["rows"]
        x = np.arange(len(rows))
        laplace = [float(row["laplace_posterior"]) for row in rows]
        quadrature = [float(row["quadrature_posterior"]) for row in rows]
        labels = [
            "CYP3A5\nnon-expresser" if row["state_id"] == "cyp3a5_nonexpresser" else "CYP3A5\nexpresser"
            for row in rows
        ]
        ax.bar(x - width / 2, laplace, width, label="Laplace package", color=colors["laplace"])
        ax.bar(x + width / 2, quadrature, width, label="Gauss-Hermite reference", color=colors["quadrature"])
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
    fig.suptitle("Tacrolimus Staatz Tier 1 two-compartment stress test", fontsize=11)
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.94))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def write_profile_plot(path: Path, result: dict[str, Any]) -> None:
    plt = _load_pyplot()
    fig, ax = plt.subplots(figsize=(8.8, 5.0), dpi=160)
    colors = {"cyp3a5_nonexpresser": "#2563eb", "cyp3a5_expresser": "#b45309"}
    for state_id in STATES:
        rows = [row for row in result["profile_rows"] if row["state_id"] == state_id]
        ax.plot(
            [float(row["time_after_dose_h"]) for row in rows],
            [float(row["concentration_ng_ml"]) for row in rows],
            marker="o",
            linewidth=2.0,
            color=colors[state_id],
            label=state_id.replace("cyp3a5_", "").replace("nonexpresser", "non-expresser"),
        )
    ax.set_xlabel("Time after dose (h)")
    ax.set_ylabel("Tacrolimus concentration (ng/mL)")
    ax.set_title("Deterministic q12h repeated-dose profiles, ETA(CL)=ETA(V1)=0")
    ax.grid(color="#d0d7de", linewidth=0.8, alpha=0.7)
    ax.legend(frameon=False)
    fig.tight_layout()
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
            label=state_id.replace("cyp3a5_", "").replace("nonexpresser", "non-expresser"),
            linewidth=2.0,
            color=colors[state_id],
        )
    for state_id, trough in curve["state_troughs_ng_ml"].items():
        ax.axvline(float(trough), color=colors[state_id], linestyle="--", linewidth=1.0, alpha=0.65)
    ax.set_xlabel(curve["x_label"])
    ax.set_ylabel("Laplace posterior probability")
    ax.set_ylim(0.0, 1.02)
    ax.set_title("Trough-only posterior curve: Tier 1 two-compartment model")
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
        / "tacrolimus_staatz_2comp_tier1_20260522"
    )
    parser = argparse.ArgumentParser(description="Run the tacrolimus Staatz Tier 1 two-compartment stress test.")
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument("--gh-nodes", type=int, default=41)
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
                "max_laplace_normalization_error": result["max_laplace_normalization_error"],
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
