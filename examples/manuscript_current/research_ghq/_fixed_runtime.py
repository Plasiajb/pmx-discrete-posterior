"""Case-local loader for the unchanged formal fixed-grid TAC source closure."""
from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from copy import deepcopy
from importlib import import_module, util
from pathlib import Path
import sys
from types import ModuleType
from uuid import uuid4


@contextmanager
def fixed_worker(policy_overrides: Mapping[str, object]) -> Iterator[ModuleType]:
    """Yield a fresh private worker; remove its entire namespace on every exit.

    The caller owns profile/input/source pin validation and supplies its pinned
    case-level overrides: maximum_order is 47 or 49 and validation_rounds is 1.
    All other policy fields, including thresholds, remain at formal source
    values. This loader checks that exact shape, not profile eligibility.

    Run case_attempt(..., integration_plan=plan) and then
    validate_result(persisted_result, case, plan) inside this context. A case
    from the existing resolved DTO is compatible; the private resolved reader
    is also available via import_module(worker.__package__ + '.resolved').
    Neither worker nor its private modules should be used after context exit.

    The formal worker's process-wide thread-environment and bytecode settings
    are retained verbatim. Module/policy isolation is not process isolation.
    """
    if not isinstance(policy_overrides, Mapping):
        raise TypeError('policy_overrides must be a mapping')
    overrides = deepcopy(dict(policy_overrides))
    if (set(overrides) != {'maximum_order', 'validation_rounds'}
            or type(overrides['maximum_order']) is not int
            or overrides['maximum_order'] not in (47, 49)
            or type(overrides['validation_rounds']) is not int
            or overrides['validation_rounds'] != 1):
        raise ValueError('Fixed-grid overrides require maximum_order 47 or 49 and validation_rounds 1 only')
    source = Path(__file__).resolve().parent / '_tac_fixed'
    namespace = f'{__package__}._tac_fixed_{uuid4().hex}'
    spec = util.spec_from_file_location(
        namespace, source / '__init__.py', submodule_search_locations=[str(source)])
    if spec is None or spec.loader is None:
        raise ImportError('Cannot load the formal fixed-grid TAC package')
    package = util.module_from_spec(spec)
    try:
        sys.modules[namespace] = package
        spec.loader.exec_module(package)
        # Import worker first to retain its original pre-NumPy thread setup.
        worker = import_module('.worker', namespace)
        import_module('.planned_grid', namespace)
        import_module('.resolved', namespace)
        worker.POLICY.update(overrides)
        yield worker
    finally:
        for name in tuple(sys.modules):
            if name == namespace or name.startswith(namespace + '.'):
                sys.modules.pop(name, None)
