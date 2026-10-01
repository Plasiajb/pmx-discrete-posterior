from __future__ import annotations
import csv, json, math, re
from pathlib import Path
TOL=1e-10
NUM='[+-]?(?:\\d+(?:\\.\\d*)?|\\.\\d+)(?:[eEdD][+-]?\\d+)?'

def require(ok, message):
    if not ok:
        raise ValueError(message)

def native_table(path):
    result = {'header': [], 'rows': [], 'raw_rows': [], 'errors': []}
    try:
        lines = Path(path).read_text(encoding='utf-8-sig', errors='strict').splitlines()
    except (OSError, UnicodeError) as exc:
        result['errors'].append(str(exc))
        return result
    for number, line in enumerate(lines, 1):
        cells = line.split()
        if not cells or cells[0] == 'TABLE':
            continue
        if cells[0] == 'SUBJECT_NO':
            if result['header']:
                result['errors'].append(f'duplicate header at {number}')
            result['header'] = cells
            if len(set(cells)) != len(cells):
                result['errors'].append('duplicate columns')
            continue
        if not result['header'] or len(cells) != len(result['header']):
            result['errors'].append(f'malformed row at {number}')
            continue
        try:
            result['rows'].append(dict(zip(result['header'], [float(v.replace('D', 'E').replace('d', 'e')) for v in cells])))
            result['raw_rows'].append(dict(zip(result['header'], cells)))
        except ValueError:
            result['errors'].append(f'nonnumeric row at {number}')
    if not result['rows']:
        result['errors'].append('missing rows')
    return result

def state_identity(table, count):
    return not table['errors'] and len(table['rows']) == count and ([r.get('SUBPOP') for r in table['rows']] == list(range(1, count + 1))) and all((r.get('ID') == 1 and r.get('SUBJECT_NO') == 1 for r in table['rows']))

def native_gate(phm, shm, states):
    details = {}
    for name, table in (('phm', phm), ('shm', shm)):
        ident = state_identity(table, len(states))
        values = [r.get('PMIX', math.nan) for r in table['rows']]
        finite_range = all((math.isfinite(v) and 0 <= v <= 1 for v in values))
        total = math.fsum(values) if finite_range else None
        zero_ok = ident and all((s['prior'] != 0 or values[i] == 0 for i, s in enumerate(states)))
        details[name] = {'identity': ident, 'finite_range': finite_range, 'sum': total, 'sum_error': abs(total - 1) if total is not None else None, 'structural_zeros': zero_ok, 'valid': ident and finite_range and (abs(total - 1) <= TOL) and zero_ok}
    parity = state_identity(phm, len(states)) and state_identity(shm, len(states))
    differences = [abs(a.get('PMIX', math.nan) - b.get('PMIX', math.nan)) for a, b in zip(phm['rows'], shm['rows'])]
    parity = parity and all((math.isfinite(d) and d <= TOL for d in differences))
    return {**details, 'parity': parity, 'parity_max': max(differences) if differences and all((math.isfinite(d) for d in differences)) else None, 'eligible': details['phm']['valid'] and details['shm']['valid'] and parity, 'tolerance': TOL}

def common_c(table, count):
    require(state_identity(table, count), 'C source state identity/parse failure')
    values = [r.get('OBJ', math.nan) for r in table['rows']]
    require(all((math.isfinite(v) for v in values)), 'C requires every expected finite native OBJ')
    return min(values)

def inspect_control(control):
    code = '\n'.join((line.split(';')[0].strip() for line in control.splitlines()))
    require(len(re.findall('^Y=IPRED\\+EPS\\(1\\)$', code, re.M)) == 1, 'Unsupported Gaussian observation pattern')
    require(len(re.findall('\\bY\\s*=', code)) == 1, 'Additional Y assignment')
    require(re.findall('EPS\\((\\d+)\\)', code) == ['1'], 'Unsupported EPS dimension/use')
    require(set(re.findall('\\bETA\\((\\d+)\\)', code)) == {'1'}, 'Requires one ETA')
    require(not re.search('F_FLAG|POPIND', code), 'Already patched or conflicting population flag')
    records = re.findall('^\\$(\\w+)([^\\n]*)', code, re.M)
    names = [k for k, _ in records]
    require(set(names) <= {'PROBLEM', 'INPUT', 'DATA', 'PRED', 'MIX', 'THETA', 'OMEGA', 'SIGMA', 'ESTIMATION', 'TABLE'}, 'Unsupported NONMEM records')
    for name in ('PROBLEM', 'INPUT', 'DATA', 'PRED', 'MIX', 'OMEGA', 'SIGMA', 'ESTIMATION', 'TABLE'):
        require(names.count(name) == 1, 'Expected one ' + name)
    params = {}
    for kind in ('THETA', 'OMEGA', 'SIGMA'):
        values = [rest.strip() for name, rest in records if name == kind]
        require(values and all((re.fullmatch(NUM + '\\s+FIX', v, re.I) for v in values)), 'Parameters must be scalar FIX: ' + kind)
        params[kind] = [float(v.split()[0].replace('D', 'E')) for v in values]
        require(all((math.isfinite(v) for v in params[kind])), 'Nonfinite fixed parameter')
    require(params['OMEGA'][0] > 0 and params['SIGMA'][0] > 0, 'Positive variance required')
    est = next((rest for name, rest in records if name == 'ESTIMATION'))
    for token in ('METHOD=1', 'INTERACTION', 'LAPLACIAN', 'POSTHOC', 'MAXEVAL=0'):
        require(len(re.findall('(?<!\\S)' + re.escape(token) + '(?!\\S)', est)) == 1, 'Estimation option mismatch: ' + token)
    require(not re.search('\\b(?:MCETA|SEED|NUMERICAL|SLOW)\\s*=', est), 'Unreviewed estimation option')
    data = next((rest.strip() for name, rest in records if name == 'DATA'))
    match = re.fullmatch('([A-Za-z0-9_.-]+)\\s+IGNORE=@', data)
    require(match is not None, 'DATA must be an isolated local filename with IGNORE=@')
    require(next((rest.strip() for name, rest in records if name == 'INPUT')) == 'ID TIME DV MDV WT AGE HT', 'INPUT mismatch')
    count = int(re.search('^\\$MIX NSPOP=(\\d+)$', code, re.M)[1])
    priors = re.findall('^P\\((\\d+)\\)=(' + NUM + ')$', code, re.M)
    require([int(k) for k, _ in priors] == list(range(1, count + 1)), 'Prior order/count mismatch')
    weights = [float(v) for _, v in priors]
    require(all((math.isfinite(p) and 0 <= p <= 1 for p in weights)) and abs(math.fsum(weights) - 1) <= TOL, 'Invalid priors')
    return {'parameters': params, 'variance': params['SIGMA'][0], 'data_name': match[1], 'state_count': count, 'priors': weights, 'estimation': est.strip()}

def transform(control, offset, nobs):
    contract = inspect_control(control)
    require(math.isfinite(offset) and isinstance(nobs, int) and (nobs > 0), 'Invalid common offset/nobs')
    variance = format(contract['variance'], '.17g')
    shift = format(offset / nobs, '.17g')
    layer = f'POPIND=EPS(1)\nF_FLAG=2\nY=LOG({variance})+(DV-IPRED)**2/({variance})-({shift})'
    updated = control.replace('Y=IPRED+EPS(1)', layer)

    def options(match):
        line = re.sub('\\s+(?:OPTMAP|FORMAT)\\s*=\\s*\\S+', '', match[0], flags=re.I)
        return line + (' OPTMAP=1' if line.startswith('$ESTIMATION') else '') + ' FORMAT=s1PE24.16'
    updated = re.sub('^\\$(?:ESTIMATION|TABLE)[^\\r\\n]*', options, updated, flags=re.M)
    restored = updated.replace(layer, 'Y=IPRED+EPS(1)')
    for prefix in ('$ESTIMATION', '$TABLE'):
        old = next((line for line in control.splitlines() if line.startswith(prefix)))
        restored = re.sub('^' + re.escape(prefix) + '[^\\r\\n]*', lambda _: old, restored, flags=re.M)
    require(restored == control, 'Unexpected transformation outside approved scope')
    return updated
