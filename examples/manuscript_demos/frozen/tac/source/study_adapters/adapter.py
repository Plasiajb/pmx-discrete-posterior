"""Project-authored executed numerical definitions; portable source projection."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import importlib
import math
import sys
import numpy as np
from scipy.linalg import expm

ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = ROOT.parent / 'pmx_discrete_posterior_v0.1.4'

def public_package():
    """Import the assigned released source read-only, rejecting another installation."""
    sys.dont_write_bytecode = True
    src = SOURCE_ROOT / "src"
    if not src.is_dir():
        raise FileNotFoundError(src)
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    pmx = importlib.import_module("pmx_discrete_posterior")
    for name, module in list(sys.modules.items()):
        if name == "pmx_discrete_posterior" or name.startswith("pmx_discrete_posterior."):
            path = getattr(module, "__file__", None)
            if path is None or not Path(path).resolve().is_relative_to(src.resolve()):
                raise RuntimeError(f"A different package module is already imported: {name}")
    return pmx


@dataclass(frozen=True)
class ModelConfig:
    cl_f_l_h: float = 25.5
    v1_f_l: float = 113.0
    q_f_l_h: float = 67.9
    v2_f_l: float = 1060.0
    ka_h: float = 0.35
    lag_h: float = 0.44
    prior_dose_mg: float = 5.0
    prior_count: int = 120
    prior_tau_h: float = 12.0
    history_anchor_h: float = 0.0
    recent_times_h: tuple[float, ...] = (0.0, 12.0)
    recent_amounts_mg: tuple[float, ...] = (5.0, 5.0)
    observation_times_h: tuple[float, ...] = (24.0,)

    def __post_init__(self):
        for key in ("recent_times_h", "recent_amounts_mg", "observation_times_h"):
            object.__setattr__(self, key, tuple(float(x) for x in getattr(self, key)))
        positive = (self.cl_f_l_h, self.v1_f_l, self.q_f_l_h, self.v2_f_l,
                    self.ka_h, self.prior_tau_h)
        if not all(math.isfinite(x) and x > 0 for x in positive):
            raise ValueError("Structural parameters and prior interval must be positive finite")
        if not all(math.isfinite(x) and x >= 0 for x in
                   (self.lag_h, self.prior_dose_mg, *self.recent_amounts_mg)):
            raise ValueError("Lag and dose amounts must be nonnegative finite")
        if isinstance(self.prior_count, bool) or not isinstance(self.prior_count, int) or self.prior_count < 0:
            raise ValueError("prior_count must be a nonnegative integer")
        if not self.recent_times_h or len(self.recent_times_h) != len(self.recent_amounts_mg):
            raise ValueError("Recent times and amounts must have the same nonzero length")
        if not self.observation_times_h or not all(math.isfinite(x) for x in
                (*self.recent_times_h, *self.observation_times_h, self.history_anchor_h)):
            raise ValueError("All times must be finite, with at least one observation")
        if any(a >= b for a, b in zip(self.recent_times_h, self.recent_times_h[1:])):
            raise ValueError("Recent doses must be strictly chronological")
        if any(a >= b for a, b in zip(self.observation_times_h, self.observation_times_h[1:])):
            raise ValueError("Observation times must be strictly increasing")
        if self.prior_count and self.history_anchor_h - self.prior_tau_h >= self.recent_times_h[0]:
            raise ValueError("Prior history must end strictly before all variable recent doses")


@dataclass(frozen=True)
class DoseState:
    state_id: str
    bits: tuple[int, ...]
    prior: float

    def __post_init__(self):
        object.__setattr__(self, "bits", tuple(self.bits))
        if not self.bits or any(x not in (0, 1) for x in self.bits):
            raise ValueError("Dose bits must be binary")
        if self.state_id != "".join(str(x) for x in self.bits):
            raise ValueError("State ID must explicitly encode latest-to-older dose bits")
        if not math.isfinite(self.prior) or self.prior < 0:
            raise ValueError("State prior must be nonnegative finite")

    @property
    def chronological_bits(self):
        return tuple(reversed(self.bits))


def build_states(ids=("00", "01", "10", "11"), priors=None):
    priors = tuple(priors) if priors is not None else (1.0 / len(ids),) * len(ids)
    if len(ids) != len(priors) or len(set(ids)) != len(ids) or not ids:
        raise ValueError("Unique state IDs and matching priors required")
    states = tuple(DoseState(s, tuple(int(c) for c in s), float(p)) for s, p in zip(ids, priors))
    if not math.isclose(sum(s.prior for s in states), 1.0, abs_tol=1e-12, rel_tol=0):
        raise ValueError("Priors must already sum to one; no implicit reweighting")
    return states


class TacrolimusAdapter:
    def __init__(self, config=ModelConfig(), states=None):
        self.config = config
        self.states = build_states() if states is None else tuple(states)
        if len({s.state_id for s in self.states}) != len(self.states) or not self.states:
            raise ValueError("State identifiers must be complete and unique")
        if any(len(s.bits) != len(config.recent_times_h) for s in self.states):
            raise ValueError("Each state bit must map to one recent dose")
        if not math.isclose(sum(s.prior for s in self.states), 1.0, abs_tol=1e-12, rel_tol=0):
            raise ValueError("Priors must sum to one")
        self._states = {s.state_id: s for s in self.states}

    def public_states(self):
        pmx = public_package()
        return tuple(pmx.DiscreteState(
            state_id=s.state_id, prior=s.prior,
            label=f"Latest-to-older dose-presence bits {s.state_id}",
            metadata={"bits_latest_to_older": list(s.bits),
                      "bits_oldest_to_newest": list(s.chronological_bits),
                      "recent_times_h": list(self.config.recent_times_h),
                      "truth_label_declared": False}) for s in self.states)

    def dose_events(self, state_id):
        c, s = self.config, self._states[state_id]
        prior = [(c.history_anchor_h - j * c.prior_tau_h, c.prior_dose_mg)
                 for j in range(c.prior_count, 0, -1)]
        recent = [(t, a) for t, a, bit in zip(c.recent_times_h, c.recent_amounts_mg, s.chronological_bits) if bit]
        return tuple((t, a) for t, a in prior + recent if a > 0)

    def predictions(self, state_id, eta):
        eta = np.asarray(eta, dtype=float)
        if eta.shape != (2,):
            raise ValueError("ETA must contain ETA(CL), ETA(V1)")
        return tuple(float(x) for x in self.batch_eta(state_id, eta[None, :])[0])

    def _exponential_dose_sum(self, rate, state_id):
        """Finite causal geometric history plus explicit recent events (no SS)."""
        c = self.config
        rate = np.asarray(rate).reshape(-1, 1)
        times = np.asarray(c.observation_times_h)[None, :]
        first = np.maximum(1, np.ceil((c.history_anchor_h + c.lag_h - times) / c.prior_tau_h))
        count = np.maximum(0, c.prior_count - first + 1)
        elapsed = np.maximum(0, times - c.history_anchor_h + first * c.prior_tau_h - c.lag_h)
        result = (c.prior_dose_mg * np.exp(-rate * elapsed)
                  * (-np.expm1(-rate * c.prior_tau_h * count))
                  / (-np.expm1(-rate * c.prior_tau_h)))
        for t, a, bit in zip(c.recent_times_h, c.recent_amounts_mg, self._states[state_id].chronological_bits):
            if bit:
                elapsed = times - t - c.lag_h
                result += a * np.exp(-rate * np.maximum(elapsed, 0)) * (elapsed >= 0)
        return result

    def batch_eta(self, state_id, eta):
        """Return (n_eta, n_observations), sharing one ETA pair across observations."""
        if state_id not in self._states:
            raise KeyError(state_id)
        eta = np.asarray(eta, dtype=float)
        if eta.ndim != 2 or eta.shape[1] != 2 or not np.all(np.isfinite(eta)):
            raise ValueError("batch_eta requires finite shape (n, 2)")
        c = self.config
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            cl, v1 = c.cl_f_l_h * np.exp(eta[:, 0]), c.v1_f_l * np.exp(eta[:, 1])
            k10, k12, k21 = cl / v1, c.q_f_l_h / v1, c.q_f_l_h / c.v2_f_l
            delta = np.hypot(k10 + k12 - k21, 2 * np.sqrt(k12 * k21))
            alpha = (k10 + k12 + k21 + delta) / 2
            beta = k10 * k21 / alpha
            near = (np.isclose(alpha, beta, rtol=1e-5, atol=1e-12)
                    | np.isclose(alpha, c.ka_h, rtol=1e-5, atol=1e-12)
                    | np.isclose(beta, c.ka_h, rtol=1e-5, atol=1e-12))
            output = np.empty((len(eta), len(c.observation_times_h)))
            ok = ~near
            a, b = alpha[ok, None], beta[ok, None]
            fa = self._exponential_dose_sum(alpha[ok], state_id)
            fb = self._exponential_dose_sum(beta[ok], state_id)
            fk = self._exponential_dose_sum(np.full(np.sum(ok), c.ka_h), state_id)
            output[ok] = 1000 * c.ka_h / v1[ok, None] * (
                (a - k21) / (a - b) * (fa - fk) / (c.ka_h - a)
                + (k21 - b) / (a - b) * (fb - fk) / (c.ka_h - b))
        for i in np.flatnonzero(near):
            output[i] = self._expm_predictions(state_id, cl[i], v1[i])
        if not np.all(np.isfinite(output)) or np.any(output < -1e-10):
            raise FloatingPointError("Nonfinite or materially negative concentration")
        return np.maximum(output, 0.0)

    def _expm_predictions(self, state_id, cl, v1):
        c = self.config
        matrix = np.array([[-c.ka_h, 0, 0],
                           [c.ka_h, -(cl + c.q_f_l_h) / v1, c.q_f_l_h / c.v2_f_l],
                           [0, c.q_f_l_h / v1, -c.q_f_l_h / c.v2_f_l]])
        return np.array([sum(a * expm(matrix * (t - td - c.lag_h))[1, 0]
                             for td, a in self.dose_events(state_id) if t >= td + c.lag_h)
                         * 1000 / v1 for t in c.observation_times_h])


def nonmem_ready_rows(model, observations, state_id=None):
    """Export event semantics only, not a NONMEM control stream or runner.

    state_id=None emits every candidate recent event plus per-state F1 flags;
    a native MIX adapter must use those flags, not the all-present AMT alone.
    """
    y = tuple(float(x) for x in observations)
    if len(y) != len(model.config.observation_times_h) or not all(math.isfinite(x) for x in y):
        raise ValueError("Finite DV must match observation times")
    c = model.config
    common = [(c.history_anchor_h - j * c.prior_tau_h, c.prior_dose_mg, -1)
              for j in range(c.prior_count, 0, -1)]
    recent = [(t, a, i) for i, (t, a) in enumerate(zip(c.recent_times_h, c.recent_amounts_mg))]
    origin = min(0.0, *(t for t, _, _ in common + recent), *c.observation_times_h)
    rows = []
    for t, amount, index in common + recent:
        flags = {s.state_id: 1 if index < 0 else s.chronological_bits[index] for s in model.states}
        if state_id is not None and not flags[state_id]:
            continue
        rows.append(dict(ID=1, TIME=t - origin, ORIGINAL_TIME_H=t, AMT=amount, DV=0.0,
                         EVID=1, MDV=1, CMT=1, SS=0, ADDL=0, II=0,
                         RECENT_INDEX=index, F1_BY_STATE=flags))
    rows += [dict(ID=1, TIME=t - origin, ORIGINAL_TIME_H=t, AMT=0.0, DV=dv,
                  EVID=0, MDV=0, CMT=2, SS=0, ADDL=0, II=0, RECENT_INDEX=-1,
                  F1_BY_STATE={}) for t, dv in zip(c.observation_times_h, y)]
    return {"time_origin_h": origin, "lag_h": c.lag_h,
            "dose_amount_unit": "mg", "dv_unit": "ng/mL", "central_scale": "V1/1000",
            "simultaneous_order": "dose before observation; depot impulse has no instantaneous central increment",
            "rows": sorted(rows, key=lambda r: (r["TIME"], -r["EVID"]))}
