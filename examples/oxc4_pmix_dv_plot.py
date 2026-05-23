from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


SCENARIOS = ("omega00", "omega01", "omega10", "omega11")
COLORS = {
    "omega00": "#667085",
    "omega01": "#1f6feb",
    "omega10": "#c47f00",
    "omega11": "#111827",
}
LABELS = {
    "omega00": "latest missed; second-most-recent missed",
    "omega01": "latest missed; second-most-recent given",
    "omega10": "latest given; second-most-recent missed",
    "omega11": "latest given; second-most-recent given",
}


def _public_api() -> dict[str, Any]:
    import pmx_discrete_posterior as pmx

    return {
        "DiscreteState": pmx.DiscreteState,
        "Config": getattr(pmx, "LaplaceConfig", None) or getattr(pmx, "OneEtaLaplaceConfig"),
        "Model": (
            getattr(pmx, "OxcSingleEtaModel", None)
            or getattr(pmx, "OxcOneEtaDoseHistoryModel", None)
            or getattr(pmx, "OxcLaplacePk")
        ),
        "build_states": getattr(pmx, "build_oxc_dose_history_states", None),
        "compute_curve": getattr(pmx, "compute_curve", None) or getattr(pmx, "build_laplace_curve_payload"),
    }


def _parse_prior(values: list[str]) -> dict[str, float]:
    if not values:
        return {scenario: 1.0 / len(SCENARIOS) for scenario in SCENARIOS}
    prior = {scenario: 0.0 for scenario in SCENARIOS}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Prior values must use state=value syntax, got {item!r}")
        state, raw_value = item.split("=", 1)
        state = state.strip()
        if state not in prior:
            raise ValueError(f"Unknown scenario {state!r}; expected one of {', '.join(SCENARIOS)}")
        prior[state] = float(raw_value)
    total = sum(max(value, 0.0) for value in prior.values())
    if total <= 0.0:
        raise ValueError("At least one prior value must be positive.")
    return {state: max(value, 0.0) / total for state, value in prior.items()}


def build_default_model(api: dict[str, Any]):
    return api["Model"](
        dose_mg=300.0,
        interval_h=12.0,
        obs_time_h=36.0,
        weight_kg=25.0,
        steady_state_anchor_age_h=36.0,
        candidate_dose_ages_h=(12.0, 24.0),
        candidate_dose_mg=(300.0, 300.0),
    )


def build_states(api: dict[str, Any], prior: dict[str, float]):
    if api["build_states"] is not None:
        return api["build_states"](SCENARIOS, prior)
    return tuple(api["DiscreteState"](state_id=scenario, prior=prior[scenario], label=scenario) for scenario in SCENARIOS)


def build_dv_grid(model: Any, points: int) -> list[float]:
    truth = [float(model.prediction(scenario, 0.0)) for scenario in SCENARIOS]
    lower = max(0.0, min(truth) * 0.40)
    upper = max(truth) * 1.15
    count = max(int(points), 2)
    step = (upper - lower) / float(count - 1)
    grid = [lower + index * step for index in range(count)]
    return sorted({round(value, 9) for value in [*grid, *truth]})


def generate_curve(
    *,
    points: int,
    prior: dict[str, float],
    omega_variance: float,
    residual_sd: float,
) -> dict[str, Any]:
    api = _public_api()
    model = build_default_model(api)
    states = build_states(api, prior)
    config = api["Config"](omega_variance=omega_variance, residual_sd=residual_sd)
    grid = build_dv_grid(model, points)
    payload = api["compute_curve"](grid, states, model, config)
    truth = {scenario: float(model.prediction(scenario, 0.0)) for scenario in SCENARIOS}
    return {
        "payload": payload,
        "truth_dv": truth,
        "prior": prior,
        "omega_variance": float(omega_variance),
        "omega_sd": math.sqrt(float(omega_variance)),
        "residual_sd": float(residual_sd),
        "grid": grid,
        "source": "pmx_discrete_posterior_public_api",
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "point_index",
        "DV",
        "scenario_key",
        "PMIX",
        "posterior_probability",
        "prior",
        "eta_hat",
        "pred_at_eta_hat",
        "psi_hat",
        "hessian",
        "laplace_log_likelihood",
        "MAP",
        "POST_MAX",
        "source",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_plot(path: Path, rows: list[dict[str, Any]], truth_dv: dict[str, float], summary: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(10.5, 6.2), dpi=160)
    for scenario in SCENARIOS:
        subset = [row for row in rows if row["scenario_key"] == scenario]
        subset.sort(key=lambda row: float(row["DV"]))
        ax.plot(
            [float(row["DV"]) for row in subset],
            [float(row["PMIX"]) for row in subset],
            color=COLORS[scenario],
            linewidth=2.1,
            label=LABELS[scenario],
        )
        ax.axvline(truth_dv[scenario], color=COLORS[scenario], linewidth=0.9, linestyle="--", alpha=0.35)

    ax.set_title("OXC 4-class PMIX-DV curve")
    ax.set_xlabel("DV / concentration (mg/L)")
    ax.set_ylabel("Posterior probability / PMIX")
    ax.set_ylim(-0.04, 1.04)
    ax.grid(True, color="#d0d7de", linewidth=0.8, alpha=0.75)
    ax.text(
        0.012,
        0.03,
        f"Laplace over ETA(CL); omega SD={summary['omega_sd']:.5g}; residual SD={summary['residual_sd']:.5g}",
        transform=ax.transAxes,
        fontsize=8.8,
        bbox={"facecolor": "white", "edgecolor": "#d0d7de", "alpha": 0.96, "boxstyle": "round,pad=0.35"},
    )
    handles = [Line2D([0], [0], color=COLORS[scenario], lw=2.6, label=LABELS[scenario]) for scenario in SCENARIOS]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def write_outputs(result: dict[str, Any], output_dir: Path, *, make_plot: bool = True) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "oxc4_pmix_dv_curve.csv"
    png_path = output_dir / "oxc4_pmix_dv_curve.png"
    summary_path = output_dir / "summary.json"
    rows = [dict(row) for row in result["payload"]["long_rows"]]
    write_csv(csv_path, rows)
    summary = {
        "source": result["source"],
        "scenarios": list(SCENARIOS),
        "prior": result["prior"],
        "omega_variance": result["omega_variance"],
        "omega_sd": result["omega_sd"],
        "residual_sd": result["residual_sd"],
        "truth_dv": result["truth_dv"],
        "point_count": len(result["grid"]),
        "row_count": len(rows),
        "csv": str(csv_path),
        "png": str(png_path) if make_plot else "",
        "boundary_note": "PMIX values are model-conditional posterior probabilities, not observed adherence truth.",
    }
    if make_plot:
        write_plot(png_path, rows, result["truth_dv"], summary)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return {"csv": str(csv_path), "png": str(png_path) if make_plot else "", "summary": str(summary_path)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate an OXC four-class PMIX-DV curve using pmx_discrete_posterior.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "examples" / "oxc4_pmix_dv_output")
    parser.add_argument("--points", type=int, default=121, help="Number of regular DV grid points before adding truth anchors.")
    parser.add_argument("--omega-variance", type=float, default=0.012245714245884425)
    parser.add_argument("--residual-sd", type=float, default=1e-6)
    parser.add_argument("--prior", action="append", default=[], help="Scenario prior as omega00=0.25; repeat for each state.")
    parser.add_argument("--no-plot", action="store_true", help="Write CSV and summary only.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    prior = _parse_prior(args.prior)
    result = generate_curve(
        points=args.points,
        prior=prior,
        omega_variance=args.omega_variance,
        residual_sd=args.residual_sd,
    )
    outputs = write_outputs(result, args.output_dir, make_plot=not args.no_plot)
    print(json.dumps(outputs, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
