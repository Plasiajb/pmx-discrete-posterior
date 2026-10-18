"""Own fixed two-start BFGS geometry; no imported comparator modes."""
import time
import numpy as np
from scipy.optimize import minimize


def gradient(fun, x, step=2e-5):
    basis = np.eye(len(x))*step
    values = fun(np.concatenate((x+basis, x-basis)))
    return (values[:len(x)]-values[len(x):])/(2*step)


def hessian(fun, x, step):
    d = len(x)
    basis = np.eye(d)*step
    rows = [x]
    for i in range(d):
        rows.extend((x+basis[i], x-basis[i]))
    for i in range(d):
        for j in range(i):
            rows.extend((x+basis[i]+basis[j], x+basis[i]-basis[j], x-basis[i]+basis[j], x-basis[i]-basis[j]))
    values = fun(np.array(rows))
    result = np.empty((d, d))
    for i in range(d):
        result[i, i] = (values[1+2*i]-2*values[0]+values[2+2*i])/step**2
    index = 1+2*d
    for i in range(d):
        for j in range(i):
            a, b, c, e = values[index:index+4]
            result[i, j] = result[j, i] = (a-b-c+e)/(4*step**2)
            index += 4
    return result


def geometry(fun, dimension, deadline, records):
    starts = (np.zeros(dimension), .25*np.where(np.arange(dimension)%2 == 0, 1., -1.))

    def checked(x):
        if time.monotonic() >= deadline:
            raise TimeoutError('Geometry deadline')
        result = np.asarray(fun(x))
        if not np.isfinite(result).all():
            raise ValueError('Nonfinite geometry objective')
        return result

    for index, start in enumerate(starts):
        record = dict(start=start.tolist(), start_index=index, status='STARTED_NOT_COMPLETED')
        records.append(record)
        result = minimize(lambda x: float(checked(x)[0]), start, jac=lambda x: gradient(checked, x),
                          method='BFGS', options=dict(gtol=3e-6, maxiter=300))
        g = gradient(checked, result.x, 1e-5)
        coarse, fine = hessian(checked, result.x, 1e-4), hessian(checked, result.x, 5e-5)
        ec, ef = np.linalg.eigvalsh(coarse), np.linalg.eigvalsh(fine)
        logdelta = abs(np.linalg.slogdet(coarse)[1]-np.linalg.slogdet(fine)[1])
        qualified = bool(result.success and np.abs(g).max() <= 1e-4 and ec.min() > 0 and ef.min() > 0 and logdelta <= 1e-3)
        record.update(status='COMPLETED', mode=result.x.tolist(), hessian=coarse.tolist(),
                      optimizer_success=bool(result.success), message=str(result.message),
                      objective=float(result.fun), gradient_inf=float(np.abs(g).max()),
                      min_eigenvalue=float(ec.min()), fine_min_eigenvalue=float(ef.min()),
                      logdet_step_change=float(logdelta), qualified=qualified, nfev=int(result.nfev))
    selected = min(records, key=lambda r: r['objective'])
    return dict(starts=records, qualified=all(r['qualified'] for r in records),
                selected_start=selected['start_index'], mode=selected['mode'], hessian=selected['hessian'])


def eigen_rotation(curvature):
    """Map Cholesky GHQ coordinates to fixed, descending-curvature eigenaxes."""
    curvature = np.asarray(curvature, dtype=float)
    values, vectors = np.linalg.eigh(curvature)
    if not np.isfinite(values).all() or values.min() <= 0:
        raise ValueError('Nonpositive curvature; no eigenvalue flooring')
    values, vectors = values[::-1], vectors[:, ::-1]
    for j in range(len(values)):
        if vectors[np.argmax(np.abs(vectors[:, j])), j] < 0:
            vectors[:, j] *= -1
    rotation = np.linalg.cholesky(curvature).T @ (vectors / np.sqrt(values))
    if not np.allclose(rotation.T @ rotation, np.eye(len(values)), rtol=0, atol=1e-12):
        raise ValueError('Eigenbasis orthogonality failed; no projection repair')
    return rotation
