"""Backend characterization; printed evidence is retained in the test history."""

import json

import numpy as np
from scipy.differentiate import hessian, jacobian


def test_scipy_offset_characterization():
    records = []
    for offset in (0.0, 5e7, 1e16):
        for curvature in (1.0, 3.0):
            for centered in (False, True):
                f = lambda x: (offset + curvature * x[0] ** 2 / 2) - (offset if centered else 0)
                result = hessian(f, np.zeros(1), tolerances={"atol": 1e-7, "rtol": 1e-6})
                records.append({"offset": offset, "curvature": curvature, "centered": centered,
                                "hessian": result.ddf.tolist(), "error": result.error.tolist(),
                                "success": result.success.tolist()})
    print(json.dumps(records, indent=2))
    assert np.allclose(records[0]["hessian"], [[1.0]])
    # A raw backend may report convergence after all curvature information is rounded away.
    assert records[-1]["success"] == [[True]]
    assert records[-1]["hessian"] == [[0.0]]


def test_scipy_gradient_characterization():
    for offset in (0.0, 5e7, 1e16):
        result = jacobian(lambda x: (offset + (x[0] - 6) ** 2) - offset,
                          np.array([5.0]), tolerances={"atol": 1e-8, "rtol": 1e-6})
        print(offset, result)
    assert result.df.shape == (1,)
