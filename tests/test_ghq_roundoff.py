from pmx_discrete_posterior import compute_posterior
from pmx_discrete_posterior.posterior import computation
from test_ghq_integration import run
from test_ghq_options import ghq_options
from test_ghq_posterior import arguments


def test_cancellation_cannot_claim_sub_roundoff_order_stability():
    # Exact Gaussian mgf: log E exp(10Z - 50) = 0. Final-result ULP alone misses cancellation.
    result = run(lambda x: 10*x[0]-50, [[1.]], ghq_options((112, 128, 144), log_integral_atol=1e-16))
    assert result["status"] == "failed", result
    assert result["log_marginal_likelihood"] is None
    assert "resolution" in result["failure_reason"] or "roundoff" in result["failure_reason"]


def test_posterior_resolution_checked_separately_from_log_tolerance(monkeypatch):
    # Exact P(b)=sigmoid(0.00005), not 0.5. Float64 saved log integrals lose the small contrast.
    args = arguments()
    args["priors"] = {"a": .5, "b": .5}
    args["numerical_options"] = ghq_options((12, 20, 32), log_integral_atol=10., posterior_atol=1e-6)
    def bind(**kwargs):
        return ((lambda x: -1e12) if kwargs["candidate_state"]["state_id"] == "a"
                else (lambda x: -1e12 + .01*x[0]))
    monkeypatch.setattr(computation, "_bind_likelihood", bind)
    result = compute_posterior(**args)
    assert result["status"] == "failed", result
    assert all(row["status"] == "success" for row in result["state_results"])
    assert all(row["posterior_probability"] is None for row in result["state_results"])
    assert "resolution" in result["failure_reason"] or "roundoff" in result["failure_reason"]
