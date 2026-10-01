"""Project-authored executed numerical definitions; portable source projection."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import math
import numpy as np
from adapter import ModelConfig, TacrolimusAdapter, build_states, public_package

@dataclass(frozen=True)
class ErrorConfig:
    eta_cl_cv: float = 0.295
    eta_v1_cv: float = 0.468
    correlation: float = 0.0
    proportional_sd: float = 0.183
    additive_sd: float = 0.05

    def __post_init__(self):
        if not all(math.isfinite(x) for x in asdict(self).values()):
            raise ValueError("Error parameters must be finite")
        if min(self.eta_cl_cv, self.eta_v1_cv) <= 0 or min(self.additive_sd, self.proportional_sd) < 0:
            raise ValueError("Both ETA variances positive; residual SDs nonnegative")
        if self.additive_sd == 0 and self.proportional_sd == 0:
            raise ValueError("At least one residual SD must be positive")
        if abs(self.correlation) >= 1:
            raise ValueError("Correlation must produce a positive definite covariance")

    def omega(self):
        a, b = math.log1p(self.eta_cl_cv ** 2), math.log1p(self.eta_v1_cv ** 2)
        ab = self.correlation * math.sqrt(a * b)
        return ((a, ab), (ab, b))

    def public_config(self):
        return public_package().MultiEtaLaplaceConfig(
            omega_covariance=self.omega(), additive_sd=self.additive_sd,
            proportional_sd=self.proportional_sd, eta_lower=-5.0, eta_upper=5.0,
            optimizer_method="L-BFGS-B", optimizer_maxiter=1000,
            derivative_step_min=1e-4, derivative_step_scale=1e-4,
            positive_floor=1e-12, prior_policy="structural_zero")


@dataclass(frozen=True)
class Case:
    case_id: str
    family: str
    subfamily: str
    description: str
    model_config: ModelConfig = ModelConfig()
    error_config: ErrorConfig = ErrorConfig()
    state_ids: tuple[str, ...] = ("00", "01", "10", "11")
    priors: tuple[float, ...] = (0.25, 0.25, 0.25, 0.25)
    dv_ng_ml: tuple[float, ...] = (12.0,)
    core_coverage: bool = False
    numerical_stress: bool = False
    truth_label_declared: bool = False
    prior_weights_input: tuple[float, ...] | None = None

    def __post_init__(self):
        for key in ("state_ids", "priors", "dv_ng_ml"):
            object.__setattr__(self, key, tuple(getattr(self, key)))
        raw = self.priors if self.prior_weights_input is None else tuple(self.prior_weights_input)
        weights = np.asarray(raw, dtype=float)
        if weights.shape != (len(self.state_ids),) or not np.all(np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError("Prior weights must be finite, nonnegative and state-matched")
        total = math.fsum(weights)
        if not math.isfinite(total) or total <= 0:
            raise ValueError("Positive finite prior mass required")
        object.__setattr__(self, "prior_weights_input", tuple(raw))
        object.__setattr__(self, "priors", tuple(weights / total))
        if self.truth_label_declared is not False:
            raise ValueError("These fixed-data numerical targets declare no truth label")
        if self.core_coverage and self.numerical_stress:
            raise ValueError("Extreme numerical sentinels cannot satisfy core coverage")
        if len(self.dv_ng_ml) != len(self.model_config.observation_times_h):
            raise ValueError("DV and observation times must match")
        if not np.all(np.isfinite(self.dv_ng_ml)):
            raise ValueError("Fixed DV must be finite")
        self.model()

    def model(self):
        return TacrolimusAdapter(self.model_config, build_states(self.state_ids, self.priors))

    def settings(self):
        data = asdict(self)
        data["omega_covariance"] = self.error_config.omega()
        data["eta_order"] = ["ETA(CL)", "ETA(V1)"]
        data["state_mapping"] = [
            {"state_id": s.state_id, "bits_latest_to_older": list(s.bits),
             "bits_oldest_to_newest": list(s.chronological_bits), "prior": s.prior,
             "recent_events": [{"time_h": t, "amount_mg": a, "present": bool(bit)}
                               for t, a, bit in zip(self.model_config.recent_times_h,
                                                   self.model_config.recent_amounts_mg, s.chronological_bits)]}
            for s in self.model().states]
        data["data_source"] = "preregistered fixed numerical DV; no generating state or clinical label"
        data["covariates"] = {"WT_kg": 70.0, "HAEM": 0.33, "POD_days": 22.7,
                              "PredCmax_free_nmol_l": 155.5, "centered_fixed": True,
                              "CL_multiplier": 1.0}
        data["parameter_range_basis"] = "Tier1 baseline; declared numerical sensitivity, not clinical ranges"
        data["prior_normalization"] = {"input_sum": math.fsum(self.prior_weights_input),
                                       "normalized_sum": math.fsum(self.priors),
                                       "rule": "divide all declared weights by their sum before either route"}
        return data


def case_from_settings(data):
    return Case(**{key: data[key] for key in ("case_id", "family", "subfamily", "description",
                 "state_ids", "priors", "dv_ng_ml", "core_coverage", "numerical_stress", "truth_label_declared")},
                prior_weights_input=data.get("prior_weights_input"),
                model_config=ModelConfig(**data["model_config"]), error_config=ErrorConfig(**data["error_config"]))
