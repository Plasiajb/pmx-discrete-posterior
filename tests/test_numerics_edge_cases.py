import numpy as np
import pytest

from pmx_discrete_posterior.interfaces.validation import InputError
from pmx_discrete_posterior.numerics.derivatives import gradient, hessian


@pytest.mark.parametrize("differentiate", [gradient, hessian])
def test_large_coordinate_unresolved_steps_not_accepted(differentiate):
    center = 1e18
    result = differentiate(lambda x: (x[0]-center)**2, [center])
    assert result["status"] == "failed", result


@pytest.mark.parametrize("differentiate", [gradient, hessian])
@pytest.mark.parametrize("step", [1e-200, 1e200])
def test_extreme_positive_steps_fail_explicitly(differentiate, step):
    try:
        result = differentiate(lambda x: x[0]**2, [1], {"backend_options": {"initial_step": step}})
    except InputError:
        return
    assert result["status"] == "failed", result


def test_asymmetric_noise_not_symmetrized_into_valid_matrix():
    result = hessian(lambda x: np.round(np.exp(x[0]+x[1]), 3), [0, 0])
    assert result["status"] == "failed", result


def test_independent_audit_offset_curvature_0414():
    result = hessian(lambda x: 5e7 + .414*x[0]**2/2, [0.])
    if result["status"] == "success":
        np.testing.assert_allclose(result["hessian"], [[.414]], atol=1e-6, rtol=1e-6)
    else:
        assert result["hessian"] is None and result["failure_reason"]


def test_finite_extreme_hessian_not_overflowed_by_symmetrization():
    result = hessian(lambda x: 5e307*x[0]**2, [0.], {"backend_options": {"initial_step": 1e-3}})
    if result["status"] == "success":
        assert np.isfinite(result["hessian"]).all(), result
        np.testing.assert_allclose(result["hessian"], [[1e308]], atol=0, rtol=1e-6)
    else:
        assert result["hessian"] is None and result["failure_reason"]


@pytest.mark.parametrize("differentiate", [gradient, hessian])
def test_near_one_step_factor_never_leaks_backend_exception(differentiate):
    try:
        result = differentiate(lambda x: x[0]**2, [0.],
                               {"backend_options": {"step_factor": np.nextafter(1., 2.)}})
    except InputError:
        return
    assert result["status"] == "failed", result


@pytest.mark.parametrize("curvature", np.geomspace(1e-3, 1e3, 12))
@pytest.mark.parametrize("offset", [0., 5e7, 1e12])
def test_curvature_scale_sweep_no_false_success(curvature, offset):
    result = hessian(lambda x: offset + .5*curvature*x[0]**2, [.231])
    if result["status"] == "success":
        np.testing.assert_allclose(result["hessian"], [[curvature]], atol=1e-6, rtol=1e-6)
    else:
        assert offset != 0, result
        assert result["hessian"] is None and result["failure_reason"]


def test_nc02_explicitly_resolved_step_preserves_probability():
    from test_numerics_hessian_adversarial import _nc02_objective, _equal_prior_posterior
    objectives = [_nc02_objective(state, True) for state in ("A", "B")]
    # Large steps are safe for this analytic quadratic, not an automatic PK retry rule.
    results = [hessian(f, [0.], {"backend_options": {"initial_step": 32.}}) for f in objectives]
    assert all(r["status"] == "success" for r in results), results
    values = [r["hessian"][0, 0] for r in results]
    posterior = _equal_prior_posterior(objectives, values)
    np.testing.assert_allclose(posterior, [.5, .5], atol=1e-6, rtol=0)
    print("NC02 explicit initial_step=32 H=", values, "posterior=", posterior.tolist())
