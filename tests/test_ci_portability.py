"""CI harness contracts without installing dependencies or running long references."""

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / ".github/scripts/test_installed.py"
if not SCRIPT.exists():
    pytest.skip("CI controls are supplied by the checkout, not the sdist", allow_module_level=True)
SPEC = importlib.util.spec_from_file_location("ci_installed", SCRIPT)
ci = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ci)


def put(root, relative, contents="pass\n"):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")
    return path


def source_tree(root):
    put(root, "pyproject.toml", '[project]\nname = "pmx-discrete-posterior"\nversion = "0.2.0"\n')
    put(root, "src/pmx_discrete_posterior/__init__.py")
    put(root, "tests/conftest.py")
    put(root, "tests/fixtures/OXC_C003.json", "{}\n")
    put(root, "examples/reference_models.py")
    put(root, "examples/posterior_reference.py")
    put(root, "tools/verify_reference.py")
    return root


def wheel_for(source, archive):
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "w") as bundle:
        for path in (source / "src").rglob("*.py"):
            bundle.write(path, path.relative_to(source / "src").as_posix())
    return archive


@pytest.mark.parametrize("platform,relative", [("nt", "Scripts/python.exe"), ("posix", "bin/python")])
def test_interpreter_layout(tmp_path, platform, relative):
    assert ci.python_in(tmp_path, platform) == tmp_path / relative


def test_environment_removes_source_injection_and_keeps_outputs_local(tmp_path):
    original = {"PYTHONPATH": "old/src", "PYTHONHOME": "old", "PYTHONUSERBASE": "user", "PATH": "bin"}
    environment = ci.clean_environment(tmp_path, original)
    assert all(key not in environment for key in ("PYTHONPATH", "PYTHONHOME", "PYTHONUSERBASE"))
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert environment["PATH"] == "bin" and original["PYTHONPATH"] == "old/src"
    for key in ("TEMP", "TMP", "TMPDIR", "PIP_CACHE_DIR"):
        assert Path(environment[key]).is_relative_to(tmp_path)


@pytest.mark.parametrize("distribution,suffix", [("wheel", ".whl"), ("sdist", ".tar.gz")])
def test_archive_selection_is_unambiguous(tmp_path, distribution, suffix):
    with pytest.raises(ValueError, match="exactly one"):
        ci.select_archive(tmp_path, distribution)
    archive = put(tmp_path, "first" + suffix)
    assert ci.select_archive(tmp_path, distribution) == archive
    put(tmp_path, "second" + suffix)
    with pytest.raises(ValueError, match="exactly one"):
        ci.select_archive(tmp_path, distribution)


def test_wheel_must_match_source_examples_checkout(tmp_path):
    source = source_tree(tmp_path / "source")
    archive = wheel_for(source, tmp_path / "package.whl")
    ci.verify_source(archive, source, "wheel")
    put(source, "src/pmx_discrete_posterior/__init__.py", "changed = True\n")
    with pytest.raises(ValueError, match="matching source examples"):
        ci.verify_source(archive, source, "wheel")


def test_source_version_stays_fixed(tmp_path):
    source = source_tree(tmp_path / "source")
    put(source, "pyproject.toml", '[project]\nname = "pmx-discrete-posterior"\nversion = "0.3.0"\n')
    with pytest.raises(ValueError, match="0.2.0"):
        ci.verify_source(tmp_path / "unused.tar.gz", source, "sdist")


def test_source_edits_and_module_removal_pass_after_rebuilding_wheel(tmp_path):
    source = source_tree(tmp_path / "source")
    removed = put(source, "src/pmx_discrete_posterior/numerics/ghq.py")
    stale = wheel_for(source, tmp_path / "stale.whl")
    removed.unlink()
    put(source, "src/pmx_discrete_posterior/__init__.py", "current_api = 'laplace'\n")
    with pytest.raises(ValueError, match="matching source examples"):
        ci.verify_source(stale, source, "wheel")
    rebuilt = wheel_for(source, tmp_path / "rebuilt.whl")
    ci.verify_source(rebuilt, source, "wheel")


def test_windows_built_wheel_matches_linux_checkout_newlines(tmp_path):
    source = source_tree(tmp_path / "source")
    module = source / "src/pmx_discrete_posterior/__init__.py"
    module.write_bytes(b"value = 1\r\n")
    archive = wheel_for(source, tmp_path / "package.whl")
    module.write_bytes(b"value = 1\n")
    ci.verify_source(archive, source, "wheel")


def test_nested_inputs_staged_without_historical_production_source(tmp_path):
    source = source_tree(tmp_path / "source")
    included = ["examples/manuscript_current/inputs/OXC_C003.json",
                "examples/manuscript_current/catalog.json", "examples/scenario_inputs.py",
                "examples/posterior_scenario.py", "tests/fixtures/nested/adapter.py"]
    excluded = ["examples/manuscript_demos/frozen/oxc/vendor/pmx_discrete_posterior/oxc.py",
                "examples/posterior_aghq.py", "examples/manuscript_aghq/bridge.R",
                "examples/manuscript_aghq/config.json", "examples/manuscript_aghq/frozen/table.json",
                "examples/manuscript_aghq/nested/reader.py", "examples/other/bridge.R",
                "examples/manuscript_demos/cli.py", "examples/manuscript_demos/data.json",
                "examples/manuscript_current/model_adapters/oxc_model.py",
                "examples/manuscript_current/cli.py", "examples/manuscript_current/inputs/legacy.py",
                "examples/manuscript_current/old_results.json",
                "examples/other/src/pmx_discrete_posterior/__init__.py",
                "examples/other/vendor/old.py", "examples/pmx_discrete_posterior/__init__.py",
                "examples/demo_output/result.json", "tests/_tmp_previous/result.json",
                "examples/__pycache__/cache.pyc"]
    for relative in included + excluded:
        put(source, relative)
    harness = tmp_path / "run"
    manifest = ci.stage_harness(source, harness)
    assert set(included) <= manifest.keys()
    assert not any(relative in manifest for relative in excluded)
    assert not (harness / "src").exists()
    for relative, digest in manifest.items():
        assert digest == hashlib.sha256((harness / relative).read_bytes()).hexdigest()
    assert (harness / ".github/scripts/test_installed.py").exists()
    assert (harness / "pyproject.toml").read_bytes() == (source / "pyproject.toml").read_bytes()


def test_sdist_uses_its_own_examples(tmp_path):
    source = source_tree(tmp_path / "source")
    example = put(source, "examples/posterior_reference.py", "from_archive = True\n")
    archive = tmp_path / "package.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        bundle.add(source, arcname="pmx-discrete-posterior-0.2.0")
    extracted = ci.extract_source(archive, tmp_path / "extracted")
    ci.verify_source(archive, extracted, "sdist")
    ci.stage_harness(extracted, tmp_path / "run")
    assert (tmp_path / "run/examples/posterior_reference.py").read_bytes() == example.read_bytes()


@pytest.mark.parametrize("unsafe", ["traversal", "link", "multiple-roots"])
def test_sdist_rejects_unsafe_or_ambiguous_archive(tmp_path, unsafe):
    archive = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        member = tarfile.TarInfo("../escape.txt" if unsafe == "traversal" else "one/link")
        if unsafe == "link":
            member.type = tarfile.SYMTYPE
            member.linkname = "../../outside"
        bundle.addfile(member, io.BytesIO())
        if unsafe == "multiple-roots":
            bundle.addfile(tarfile.TarInfo("two/file"), io.BytesIO())
    with pytest.raises((ValueError, tarfile.FilterError)):
        ci.extract_source(archive, tmp_path / "extracted")
    assert not (tmp_path / "escape.txt").exists()


@pytest.mark.parametrize("selection,reference", [("quick", "OXC"), ("slow", "TAC")])
def test_commands_keep_installed_checks_and_actual_references(tmp_path, selection, reference):
    tests, example = ci.check_commands(tmp_path / "python", tmp_path / "run", tmp_path, selection, reference)
    assert tests[tests.index("-m") + 1] == "pytest"
    assert tests[tests.index("--confcutdir") + 1] == "tests"
    assert tests[tests.index("-c") + 1] == str(tmp_path / "run/pyproject.toml")
    assert "pythonpath=" in tests and "--require-installed" in tests
    assert tests[tests.index("-m", tests.index("-m") + 1) + 1] == selection
    assert example[-3:] == ["tools/verify_reference.py", reference, "--require-installed"]


def test_import_probe_rejects_local_source_shadowing(tmp_path):
    put(tmp_path, "pmx_discrete_posterior/__init__.py")
    result = subprocess.run([sys.executable, "-B", "-c", ci.IMPORT_PROBE], cwd=tmp_path,
                            env=ci.clean_environment(tmp_path), capture_output=True, text=True)
    assert result.returncode != 0
    assert "Source checkout masks installed package" in result.stderr


def test_subprocess_failure_is_not_hidden_and_json_stdout_stays_clean(tmp_path):
    log = tmp_path / "reference.json"
    errors = tmp_path / "reference.stderr.log"
    code = "import sys; print('{}'); print('diagnostic', file=sys.stderr); sys.exit(7)"
    with pytest.raises(subprocess.CalledProcessError) as failure:
        ci.run([sys.executable, "-B", "-c", code], tmp_path, ci.clean_environment(tmp_path), log, stderr=errors)
    assert failure.value.returncode == 7
    assert json.loads(log.read_text(encoding="utf-8")) == {}
    assert "diagnostic" in errors.read_text(encoding="utf-8")


def test_workflow_preserves_windows_and_adds_linux_quick_oxc():
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    windows = workflow.split("  portable:\n", 1)[1].split("  tac-reference:\n", 1)[0]
    tac = workflow.split("  tac-reference:\n", 1)[1].split("  linux-installed:\n", 1)[0]
    linux = workflow.split("  linux-installed:\n", 1)[1]
    assert "runs-on: windows-latest" in windows and "distribution: [wheel, sdist]" in windows
    assert "-Selection quick -Reference OXC" in windows
    assert "runs-on: windows-latest" in tac and "-Selection slow -Reference TAC" in tac
    assert "Clean-installed TAC Laplace and curvature regressions" in tac
    assert "bounded GHQ" not in workflow
    assert "runs-on: ubuntu-latest" in linux and "distribution: [wheel, sdist]" in linux
    assert 'python: "3.11"' in linux and 'python: "3.12"' in linux
    assert "dependencies: minimum" in linux and "dependencies: current" in linux
    assert "--selection quick --reference OXC" in linux and "--reference TAC" not in linux
    assert "if: always()" in linux and "*-quick/results/*" in linux


def test_windows_wrapper_uses_shared_harness_and_propagates_exit_status():
    wrapper = (ROOT / ".github/scripts/test-installed.ps1").read_text(encoding="utf-8")
    assert "test_installed.py" in wrapper
    for argument in ("Distribution", "Dependencies", "Selection", "Reference"):
        assert f"--{argument.lower()} ${argument}" in wrapper
    assert "exit $LASTEXITCODE" in wrapper


@pytest.mark.parametrize("distribution", ["wheel", "sdist"])
@pytest.mark.parametrize("explicit", [False, True])
def test_clean_run_records_artifacts_and_uses_matching_staged_source(tmp_path, monkeypatch, distribution, explicit):
    source = source_tree(tmp_path / "repo")
    dist = source / ".github/_ci/dist"
    dist.mkdir(parents=True)
    if distribution == "wheel":
        archive = wheel_for(source, dist / "package.whl")
    else:
        archive = dist / "package.tar.gz"
        with tarfile.open(archive, "w:gz") as bundle:
            for relative in ("pyproject.toml", "src", "tests", "examples", "tools"):
                bundle.add(source / relative, arcname="package/" + relative)
    calls = []

    def fake_run(command, cwd, env, log, stderr=None):
        calls.append((command, cwd, env))
        log.write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(ci, "ROOT", source)
    monkeypatch.setattr(ci, "run", fake_run)
    arguments = ["--distribution", distribution, "--dependencies", "minimum",
                 "--selection", "quick", "--reference", "OXC"]
    if explicit:
        work = tmp_path / "external-work"
        arguments += ["--archive", str(archive), "--work-dir", str(work)]
        put(dist, "unrelated.whl" if distribution == "wheel" else "unrelated.tar.gz")
    assert ci.main(arguments) == 0
    if not explicit:
        work, = (source / ".github/_ci").glob(f"{distribution}-minimum-*-quick")
    results = work / "results"
    selection = json.loads((results / "selection.json").read_text(encoding="utf-8"))
    assert selection == {"distribution": distribution, "dependencies": "minimum", "selection": "quick",
                         "reference": "OXC", "archive": str(archive.resolve()), "work_dir": str(work.resolve())}
    assert isinstance(selection["archive"], str) and isinstance(selection["work_dir"], str)
    artifact = json.loads((results / "distribution-sha256.json").read_text())
    assert artifact == {"Artifact": archive.name, "Hash": ci.sha256(archive)}
    inputs = json.loads((results / "staged-inputs-sha256.json").read_text())
    assert "examples/posterior_reference.py" in inputs["files"]
    assert (Path(inputs["source"]) == source) == (distribution == "wheel")
    assert (results / "versions.txt").exists() and (results / "installed-import.json").exists()
    commands = [command for command, _, _ in calls]
    assert commands[0][-2:] == ["venv", str(work / "venv")]
    assert str(archive) in commands[1] and "numpy==1.26.0" in commands[1]
    assert commands[-1][-3:] == ["tools/verify_reference.py", "OXC", "--require-installed"]
    assert all(cwd == work / "run" for _, cwd, _ in calls[1:])
    assert all("PYTHONPATH" not in env for _, _, env in calls)


def test_explicit_work_directory_cannot_reuse_stale_files(tmp_path):
    archive = put(tmp_path, "package.whl")
    work = tmp_path / "existing"
    sentinel = put(work, "keep.txt", "keep")
    with pytest.raises(FileExistsError):
        ci.main(["--distribution", "wheel", "--dependencies", "current", "--selection", "quick",
                 "--reference", "OXC", "--archive", str(archive), "--work-dir", str(work)])
    assert sentinel.read_text() == "keep"


@pytest.mark.parametrize("filename", ["missing.whl", "wrong.tar.gz"])
def test_explicit_archive_must_exist_and_match_distribution(tmp_path, filename):
    if filename.startswith("wrong"):
        put(tmp_path, filename)
    work = tmp_path / "work"
    with pytest.raises(SystemExit) as failure:
        ci.main(["--distribution", "wheel", "--dependencies", "current", "--selection", "quick",
                 "--reference", "OXC", "--archive", str(tmp_path / filename), "--work-dir", str(work)])
    assert failure.value.code == 2 and not work.exists()
