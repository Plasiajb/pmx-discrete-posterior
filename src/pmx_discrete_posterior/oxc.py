from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

from .api import DiscreteState


@dataclass(frozen=True)
class OxcOneEtaDoseHistoryModel:
    """Oxcarbazepine one-eta dose-history model used as a package example."""

    dose_mg: float = 300.0
    interval_h: float = 12.0
    obs_time_h: float = 36.0
    weight_kg: float = 25.0
    v_l: float = 14.7
    ka_h: float = 0.83
    theta_cl_base: float = 1.68
    theta_wt_exp_base: float = 0.624
    theta_wt_exp_max_dec: float = 0.233
    theta_wt_exp_hill: float = 2.19
    theta_wt_exp_half: float = 8.97
    steady_state_anchor_age_h: float | None = None
    candidate_dose_ages_h: tuple[float, ...] = ()
    candidate_dose_mg: tuple[float, ...] = ()
    scenario_prefix: str = "omega"

    @property
    def wt_exp(self) -> float:
        wt_hill = self.weight_kg**self.theta_wt_exp_hill
        wt_half = self.theta_wt_exp_half**self.theta_wt_exp_hill
        return self.theta_wt_exp_base - (self.theta_wt_exp_max_dec * wt_hill) / (wt_half + wt_hill)

    @property
    def cl_typical_l_h(self) -> float:
        return self.theta_cl_base * (self.weight_kg / 70.0) ** self.wt_exp

    def k_elim(self, eta_cl: float) -> float:
        return self.cl_typical_l_h * math.exp(float(eta_cl)) / self.v_l

    def oral_contribution(
        self,
        dose_age_h: float,
        eta_cl: float,
        dose_mg: float | None = None,
    ) -> float:
        dose_age_h = float(dose_age_h)
        if dose_age_h < 0:
            return 0.0
        dose = self.dose_mg if dose_mg is None else float(dose_mg)
        k = self.k_elim(eta_cl)
        ka = self.ka_h
        if abs(ka - k) < 1e-8:
            return dose * ka / self.v_l * dose_age_h * math.exp(-k * dose_age_h)
        return dose * ka / (self.v_l * (ka - k)) * (
            math.exp(-k * dose_age_h) - math.exp(-ka * dose_age_h)
        )

    def steady_state_anchor_contribution(self, eta_cl: float) -> float:
        t = (
            self.obs_time_h
            if self.steady_state_anchor_age_h is None
            else float(self.steady_state_anchor_age_h)
        )
        if t < 0:
            return 0.0
        tau = max(float(self.interval_h), 1e-12)
        k = self.k_elim(eta_cl)
        ka = self.ka_h
        if abs(ka - k) < 1e-8:
            k = k * (1.0 + 1e-8)
        denom_k = max(1.0 - math.exp(-k * tau), 1e-300)
        denom_ka = max(1.0 - math.exp(-ka * tau), 1e-300)
        return self.dose_mg * ka / (self.v_l * (ka - k)) * (
            math.exp(-k * t) / denom_k
            - math.exp(-ka * t) / denom_ka
        )

    def prediction(self, state_id: str, eta: float) -> float:
        bits = str(state_id)
        if self.scenario_prefix and bits.startswith(self.scenario_prefix):
            bits = bits[len(self.scenario_prefix) :]

        concentration = self.steady_state_anchor_contribution(float(eta))
        for index, bit in enumerate(bits):
            if bit != "1":
                continue
            age = (
                float(self.candidate_dose_ages_h[index])
                if index < len(self.candidate_dose_ages_h)
                else float(index + 1) * self.interval_h
            )
            dose = (
                float(self.candidate_dose_mg[index])
                if index < len(self.candidate_dose_mg)
                else self.dose_mg
            )
            concentration += self.oral_contribution(age, float(eta), dose)
        return max(float(concentration), 0.0)


def build_oxc_dose_history_states(
    scenarios: Sequence[str],
    prior: Mapping[str, float] | None = None,
    *,
    candidate_event_labels: Sequence[str] | None = None,
    candidate_event_times_h: Sequence[float] | None = None,
    candidate_dose_mg: Sequence[float] | None = None,
    bit_order: str = "latest_to_older",
) -> tuple[DiscreteState, ...]:
    """Build DiscreteState entries for OXC binary dose-history scenarios."""

    if prior is None:
        prior_value = 1.0 / max(len(scenarios), 1)
        prior = {str(scenario): prior_value for scenario in scenarios}
    max_bits = max((len(str(scenario).removeprefix("omega")) for scenario in scenarios), default=0)
    labels = tuple(candidate_event_labels or _default_candidate_labels(max_bits, bit_order=bit_order))
    return tuple(
        DiscreteState(
            state_id=str(scenario),
            prior=float(prior.get(str(scenario), 0.0)),
            label=_scenario_label(str(scenario), labels),
            metadata={
                "definition": "OXC dose-history scenario",
                "bit_order": bit_order,
                "candidate_events": _scenario_assignments(
                    str(scenario),
                    labels,
                    candidate_event_times_h=candidate_event_times_h,
                    candidate_dose_mg=candidate_dose_mg,
                ),
                **_flat_assignment_metadata(str(scenario), labels),
            },
        )
        for scenario in scenarios
    )


OxcLaplacePk = OxcOneEtaDoseHistoryModel


def _default_candidate_labels(count: int, *, bit_order: str) -> tuple[str, ...]:
    if count == 2 and bit_order == "latest_to_older":
        return ("latest_24h", "second_most_recent_12h")
    return tuple(f"candidate_{index + 1}" for index in range(count))


def _scenario_bits(state_id: str) -> str:
    return str(state_id).removeprefix("omega")


def _scenario_assignments(
    state_id: str,
    labels: Sequence[str],
    *,
    candidate_event_times_h: Sequence[float] | None,
    candidate_dose_mg: Sequence[float] | None,
) -> tuple[dict[str, object], ...]:
    assignments = []
    for index, bit in enumerate(_scenario_bits(state_id)):
        label = labels[index] if index < len(labels) else f"candidate_{index + 1}"
        state = "given" if bit == "1" else "missed"
        row: dict[str, object] = {
            "bit_index": index,
            "event_label": label,
            "dose_state": state,
            "bit_value": int(bit) if bit in {"0", "1"} else bit,
        }
        if candidate_event_times_h is not None and index < len(candidate_event_times_h):
            row["event_time_h"] = float(candidate_event_times_h[index])
        if candidate_dose_mg is not None and index < len(candidate_dose_mg):
            row["planned_dose_mg"] = float(candidate_dose_mg[index])
        assignments.append(row)
    return tuple(assignments)


def _flat_assignment_metadata(state_id: str, labels: Sequence[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for index, bit in enumerate(_scenario_bits(state_id)):
        label = labels[index] if index < len(labels) else f"candidate_{index + 1}"
        out[label] = "given" if bit == "1" else "missed"
    return out


def _scenario_label(state_id: str, labels: Sequence[str]) -> str:
    parts = [
        f"{label}={'given' if bit == '1' else 'missed'}"
        for label, bit in zip(labels, _scenario_bits(state_id))
    ]
    return f"{state_id}: " + ", ".join(parts) if parts else state_id
