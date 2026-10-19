"""功能 / Function: 导出预测调用与校验函数 / Export prediction evaluation and validation functions.
输入 / Inputs: 无运行时输入 / No runtime inputs.
输出 / Outputs: 保留原导入入口，实现位于evaluation.py / Existing imports backed by evaluation.py.
"""

from ...interfaces.inputs.observations import validate_records
from ...interfaces.inputs.candidate_states import validate_candidate_history, validate_event
from ...interfaces.inputs.individual_data import validate_prediction_context
from .evaluation import evaluate_prediction, prepare_prediction_records, validate_predictions
