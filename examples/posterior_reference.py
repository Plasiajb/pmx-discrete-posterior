"""功能 / Function: 从参考输入计算完整后验 / Calculate complete posteriors from reference inputs.
输入 / Inputs: 可选OXC或TAC命令行名称及本地参考文件 / Optional OXC or TAC CLI name and local reference files.
输出 / Outputs: JSON候选概率与失败原因；进程失败时返回非零退出码 / JSON state probabilities and reasons, with a nonzero exit code on failure.
"""

import argparse
import json

from pmx_discrete_posterior import compute_posterior
from pmx_discrete_posterior.interfaces.inputs import validate_candidate_states, validate_model
from reference_models import load_reference


def main():
    """功能 / Function: 调用公开入口运行一个参考场景 / Run one reference scenario through the public entry.
    输入 / Inputs: 模型名，默认OXC / Model name, default OXC.
    输出 / Outputs: 打印每个候选的先验、积分、后验和状态；返回退出码 / Print each candidate's prior, integral, posterior and status; return an exit code.
    """
    parser = argparse.ArgumentParser(description="Laplace or prior-centered GHQ posterior for an OXC/TAC reference scenario")
    parser.add_argument("model", nargs="?", choices=("OXC", "TAC"), default="OXC")
    parser.add_argument("--method", choices=("laplace", "ghq"), default="laplace")
    parser.add_argument("--ghq-orders", nargs="+", type=int)
    parser.add_argument("--log-integral-atol", type=float)
    parser.add_argument("--posterior-atol", type=float)
    parser.add_argument("--max-nodes", type=int)
    args = parser.parse_args()
    ghq_fields = (args.ghq_orders, args.log_integral_atol, args.posterior_atol, args.max_nodes)
    options = None
    if args.method == "ghq":
        if any(value is None for value in ghq_fields):
            parser.error("GHQ requires --ghq-orders, --log-integral-atol, --posterior-atol and --max-nodes")
        options = {"ghq": {"variant": "prior_centered", "orders": args.ghq_orders,
                            "log_integral_atol": args.log_integral_atol,
                            "posterior_atol": args.posterior_atol, "max_nodes": args.max_nodes}}
    elif any(value is not None for value in ghq_fields):
        parser.error("GHQ options are not used by Laplace; select --method ghq")

    # 输入 / Inputs: 复用显式模型、观测、前史与候选定义 / Reuse explicit model, observation, prehistory and candidate definitions.
    case = load_reference(args.model)
    # 功能 / Function: 写入参考设置前复用校验 / Reuse validation before applying reference settings.
    # 输入 / Inputs: 载入的模型和候选 / Loaded model and candidates.
    # 输出 / Outputs: 校验副本或InputError；不执行模型 / Validated copies or InputError, without model execution.
    model = validate_model(case.get("model"))
    candidate_states = validate_candidate_states(case.get("candidate_states"))
    model["random_effects_dependence"] = "independent_blocks"
    model["residual_model"]["dependence"] = "conditionally_independent"
    priors = {state["state_id"]: .25 for state in candidate_states}

    # 计算 / Calculation: 七组主输入；不手写似然或再次加权 / Seven input groups, without a custom likelihood or duplicate weighting.
    result = compute_posterior(
        model=model, individual_data=case.get("individual_data"), observations=case.get("observations"),
        candidate_states=candidate_states, priors=priors,
        integration_method=args.method, numerical_options=options,
    )

    # 输出 / Output: 展示候选事件及核心结果；完整诊断仍在result中 / Show candidate events and core results; full diagnostics remain in result.
    events = {state["state_id"]: state["events"] for state in candidate_states}
    rows = [{**{k: v for k, v in row.items() if k != "method_details" or args.method == "ghq"}, "events": events[row["state_id"]]}
            for row in result["state_results"]]
    print(json.dumps({"model": args.model, **result, "state_results": rows}, indent=2, allow_nan=False))
    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
