from __future__ import annotations
import math, dataclasses, csv, io, re
import portable_support as e01
import portable_gates as c3
ADAPTER_VERSION='gamma219_stable_ratio_v1'
POWER='WT_HILL=WT**THETA(6)\nWT_HALF=THETA(7)**THETA(6)\nWTEXP=THETA(4)-(THETA(5)*WT_HILL)/(WT_HALF+WT_HILL)'
STABLE='WTLOG=THETA(6)*LOG(WT/THETA(7))\nIF (WTLOG.GE.0) THEN\n  WTRAT=1/(1+EXP(-WTLOG))\nELSE\n  WTNEG=EXP(WTLOG)\n  WTRAT=WTNEG/(1+WTNEG)\nENDIF\nWTEXP=THETA(4)-THETA(5)*WTRAT'

def ratio(wt, half, hill):
    if not all((math.isfinite(x) for x in (wt, half, hill))) or min(wt, half) <= 0:
        raise ValueError('Finite positive weights required')
    z = hill * math.log(wt / half)
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)

class StableExponent:

    @property
    def wt_exp(self):
        m = self.model
        return float(m['theta_wt_exp_base']) - float(m['theta_wt_exp_max_dec']) * ratio(float(self.patient['weight_kg']), float(m['theta_wt_exp_half']), float(m['theta_wt_exp_hill']))

class StableOxcModel(StableExponent, e01.OxcRouteModel):
    pass

class StableHistoryModel(StableExponent, e01.ExplicitHistoryOxcModel):
    pass

def model(n, stable=True):
    kwargs = dict(model=n['model'], patient=n['patient'], observations=n['observation_events'], candidate_events=e01.build_candidate_events(n), dose_interval_h=float(n['model'].get('dose_interval_h', 12.0)), steady_state_at_time_zero=bool(n['time_zero']['steady_state_at_time_zero']), time_zero_epsilon_h=float(n['time_zero']['nonmem_dose_time_zero_epsilon_h']))
    if 'e01' in n:
        h = n['e01']['history']
        kwargs.update(history_dose_mg=h['dose_mg'], history_interval_h=h['interval_h'], history_last_dose_time_h=h['last_dose_time_h'], candidate_window_start_h=h['candidate_window_start_h'])
        return (StableHistoryModel if stable else e01.ExplicitHistoryOxcModel)(**kwargs)
    return (StableOxcModel if stable else e01.OxcRouteModel)(**kwargs)

def stable_control(original, gamma=219):
    original_contract = c3.inspect_control(original)
    assert original_contract['parameters']['THETA'][5] == 2.19
    assert original.count(POWER) == 1, 'Unsupported parent power implementation'
    out = original.replace(POWER, STABLE)
    lines = out.splitlines(keepends=True)
    indexes = [i for i, line in enumerate(lines) if line.startswith('$THETA ')]
    i = indexes[5]
    assert re.match('^\\$THETA 2\\.19 FIX', lines[i]), 'Hill THETA identity'
    lines[i] = lines[i].replace('$THETA 2.19 FIX', f'$THETA {gamma:.17g} FIX', 1)
    out = ''.join(lines)
    check = c3.inspect_control(out)
    expected = original_contract['parameters'].copy()
    expected['THETA'] = list(expected['THETA'])
    expected['THETA'][5] = gamma
    assert check['parameters'] == expected
    return out

def precise_native(n, original):
    """Preserve parent equations; regenerate all model/data numbers from JSON."""
    control = stable_control(original, n['model']['theta_wt_exp_hill'])
    m, patient = (n['model'], n['patient'])
    order = n['candidate_definition']['candidate_order']
    ledger = []

    def token(value, role):
        value = float(value)
        assert math.isfinite(value)
        printed = format(value, '.17g')
        parsed = float(printed)
        assert parsed.hex() == value.hex()
        ledger.append({'role': role, 'value': value, 'token': printed, 'binary64_hex': value.hex(), 'roundtrip_exact': True})
        return printed
    keys = ('theta_cl_base_l_h', 'v_l', 'ka_h', 'theta_wt_exp_base', 'theta_wt_exp_max_dec', 'theta_wt_exp_hill', 'theta_wt_exp_half')
    values = [m[k] for k in keys]
    lines = control.splitlines()
    ix = [i for i, line in enumerate(lines) if line.startswith('$THETA ')]
    assert len(ix) == 9
    history = n.get('e01', {}).get('history')
    tau = history['interval_h'] if history else m.get('dose_interval_h', 12.0)
    dose = history['dose_mg'] if history else model(n)._anchor_dose_mg()
    assert bool(n['time_zero']['steady_state_at_time_zero'])
    for i, value in enumerate(values):
        at = ix[i]
        comment = lines[at].split(';', 1)[1]
        lines[at] = f'$THETA {token(value, 'model.' + keys[i])} FIX ;{comment}'
    for at in ix[7:]:
        label = lines[at].split(';', 1)[1]
        assert label in ('TAU_H', 'DOSE_MG', 'STEADY_STATE_DOSE_MG')
        value = tau if label == 'TAU_H' else dose
        lines[at] = f'$THETA {token(value, label)} FIX ;{label}'
    for i, line in enumerate(lines):
        if line.startswith('$OMEGA '):
            lines[i] = f'$OMEGA {token(m['iiv']['omega_variance'], 'OMEGA')} FIX ;BSV_CL'
        elif line.startswith('$SIGMA '):
            lines[i] = f'$SIGMA {token(float(m['ruv']['additive_sd_mg_l']) ** 2, 'SIGMA_SD_squared')} FIX ;ADD_VAR'
        elif re.match('^P\\(\\d+\\)=', line):
            j = int(re.search('\\d+', line)[0])
            lines[i] = f'P({j})={token(n['prior_vector'][order[j - 1]], 'prior.' + order[j - 1])}'
    control = '\n'.join(lines) + '\n'
    events = e01.build_candidate_events(n)
    if 'AGE_S1E' in control or re.search('AGE_S\\d+E\\d+', control):
        for j, state in enumerate(order, 1):
            for k, event in enumerate(events[state], 1):
                prefix = f'S{j}E{k}'
                before = f'AGE_{prefix}=TIME-({float(event['nonmem_time_h']):.8g})'
                after = f'AGE_{prefix}=TIME-({token(event['nonmem_time_h'], state + '.' + event['id'] + '.time')})'
                assert control.count(before) == 1
                control = control.replace(before, after)
                before = f'C=C+D_{prefix}*A_{prefix}*{float(event['dose_mg']):.8g}*KA/'
                after = f'C=C+D_{prefix}*A_{prefix}*{token(event['dose_mg'], state + '.' + event['id'] + '.dose')}*KA/'
                assert control.count(before) == 1
                control = control.replace(before, after)
                assert f'IF (MIXNUM.EQ.{j}) D_{prefix}=1' in control
    else:
        assert 'AGE0=TIME-0.0001' in control and 'AGE12=TIME-12' in control
        assert order == ['omega00', 'omega01', 'omega10', 'omega11']
        for j, state in enumerate(order, 1):
            expected = [e for e in n['dose_events_for_nonmem'] if float(e['time_h']) == 0 and j in (2, 4) or (float(e['time_h']) == 12 and j in (3, 4))]
            assert {e['id']: e for e in events[state]} == {e['id']: e for e in expected}
            assert all((float(e['dose_mg']) == float(dose) for e in expected))
        eps = n['time_zero']['nonmem_dose_time_zero_epsilon_h']
        assert float(eps) == 0.0001
        control = control.replace('AGE0=TIME-0.0001', 'AGE0=TIME-' + token(eps, 'dose_zero_time_epsilon'))
        token(12, 'dose_12_time')
    if history:
        before = f'TSS=TIME-({history['last_dose_time_h']:g})'
        assert control.count(before) == 1
        control = control.replace(before, f'TSS=TIME-({token(history['last_dose_time_h'], 'history.last_time')})')
    else:
        assert 'TSS=TIME+TAU' in control
    control = control.replace('$TABLE ID TIME DV MDV IPRED ', '$TABLE ID TIME DV MDV WT AGE HT IPRED ')
    rows = [['ID', 'TIME', 'DV', 'MDV', 'WT', 'AGE', 'HT']]
    expected_rows = []
    for i, observation in enumerate((o for o in n['observation_events'] if o.get('enabled', True))):
        values = [1, observation['time_h'], observation['dv_mg_l'], 0, patient['weight_kg'], patient['age_years'], patient['height_cm']]
        row = [token(v, f'observation_record_{i + 1}.{key}') for key, v in zip(rows[0], values)]
        rows.append(row)
        expected_rows.append(dict(zip(rows[0], map(float, values))))
    dataset = '\n'.join((','.join(row) for row in rows)) + '\n'
    parsed = list(csv.DictReader(io.StringIO(dataset)))
    assert all((all((float(row[k]).hex() == value.hex() for k, value in expected.items())) for row, expected in zip(parsed, expected_rows)))
    contract = c3.inspect_control(control)
    assert contract['parameters']['THETA'][:7] == [float(m[k]) for k in keys]
    assert contract['parameters']['OMEGA'] == [float(m['iiv']['omega_variance'])]
    assert contract['parameters']['SIGMA'] == [float(m['ruv']['additive_sd_mg_l']) ** 2]
    assert contract['priors'] == [float(n['prior_vector'][s]) for s in order]
    return (control, dataset, {'policy': 'roundtrip_binary64_from_current_JSON_17sig_v1', 'all_serialized_values_exact': True, 'numeric_ledger': ledger, 'expected_observation_records': expected_rows, 'event_history_state_mapping_checked': True})

def evaluate(n, route, nodes=41, stable=True):
    states, config, observations = e01.inference_inputs(n)
    order = [s.state_id for s in states]
    current = model(n, stable=stable)
    if route == 'package_laplace':
        result = e01.pmx.compute_multieta_posterior(observations, states, current, config)
        posterior = dict(result.posterior)
        diagnostics = {k: dataclasses.asdict(v) for k, v in result.diagnostics.items()}
        marginals = {k: float(v.laplace_log_likelihood) for k, v in result.diagnostics.items()}
        mode_ok = list(diagnostics) == order and all((d['valid_laplace'] and d['optimizer_success'] and math.isfinite(d['eta_hat'][0]) and math.isfinite(d['hessian_min_eigenvalue']) and (d['hessian_min_eigenvalue'] > 0) for d in diagnostics.values()))
    else:
        posterior, marginals, diagnostics = e01.adaptive_ghq_posterior(pmx=e01.pmx, observations=observations, states=states, model=current, config=config, nodes=nodes, mode_optimizer_method='Nelder-Mead')
        mode_ok = list(diagnostics) == order and all((d['valid_mode'] and d['optimizer_success'] and math.isfinite(d['eta_hat']) and math.isfinite(d['hessian']) and (d['hessian'] > 0) for d in diagnostics.values()))
    gate = e01.validity(order, list(posterior.items()))
    finite_marginals = list(marginals) == order and all((math.isfinite(v) for v in marginals.values()))
    zero_ok = all((s.prior != 0 or posterior[s.state_id] == 0 for s in states))
    eligible = gate['valid'] and mode_ok and finite_marginals and zero_ok
    return dict(posterior=posterior, log_marginals=marginals, diagnostics=diagnostics, vector_gate=gate, mode_ok=mode_ok, finite_marginals=finite_marginals, structural_zeros=zero_ok, eligible=eligible, config=dataclasses.asdict(config), candidate_order=order, adapter=ADAPTER_VERSION, package_version=e01.pmx.__version__)
