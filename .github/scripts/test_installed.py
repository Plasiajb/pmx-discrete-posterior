"""Clean wheel/sdist checks on Windows and Linux; never stage production src."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import zipfile


ROOT = Path(__file__).resolve().parents[2]
VERSION = "0.2.0"
PACKAGE = "pmx_discrete_posterior"
DEPENDENCIES = {
    "minimum": ["numpy==1.26.0", "scipy==1.16.2"],
    "current": ["numpy>=2,<3", "scipy>=1.16.2,<2"],
}
EXCLUDED = {"src", "vendor", "manuscript_demos", "manuscript_aghq", "posterior_aghq.py",
            PACKAGE, "__pycache__", ".git", ".pytest_cache"}
SUFFIXES = {".py", ".json", ".csv", ".tsv", ".md", ".txt", ".ini", ".toml"}
IMPORT_PROBE = """
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys
import sysconfig
import pmx_discrete_posterior
package = Path(pmx_discrete_posterior.__file__).resolve()
site = Path(sysconfig.get_path('purelib')).resolve()
if not package.is_relative_to(site):
    raise RuntimeError(f'Source checkout masks installed package: {package}')
if version('pmx-discrete-posterior') != '0.2.0':
    raise RuntimeError('Expected installed version 0.2.0')
print(json.dumps({'package_file': str(package), 'site_packages': str(site),
                  'version': version('pmx-discrete-posterior'),
                  'python': sys.version, 'platform': platform.platform(),
                  'sys_path': sys.path}, indent=2))
"""


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def python_in(venv, platform=None):
    return venv / ("Scripts/python.exe" if (platform or os.name) == "nt" else "bin/python")


def clean_environment(work, inherited=None):
    env = dict(os.environ if inherited is None else inherited)
    for name in ("PYTHONPATH", "PYTHONHOME", "PYTHONUSERBASE"):
        env.pop(name, None)
    temporary = work / "tmp"
    temporary.mkdir(parents=True, exist_ok=True)
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8", PYTHONNOUSERSITE="1",
               PIP_DISABLE_PIP_VERSION_CHECK="1", PIP_CACHE_DIR=str(work / "pip-cache"),
               TEMP=str(temporary), TMP=str(temporary), TMPDIR=str(temporary))
    return env


def select_archive(dist, distribution):
    archives = sorted(dist.glob("*.whl" if distribution == "wheel" else "*.tar.gz"))
    if len(archives) != 1:
        raise ValueError(f"Expected exactly one {distribution} distribution in {dist}, got {len(archives)}")
    return archives[0]


def extract_source(archive, destination):
    destination.mkdir()
    with tarfile.open(archive) as bundle:
        # Disallow links even when tarfile's data filter considers them safe.
        if any(not (member.isfile() or member.isdir()) for member in bundle.getmembers()):
            raise ValueError("Source distribution must contain only regular files and directories")
        bundle.extractall(destination, filter="data")
    roots = list(destination.iterdir())
    if len(roots) != 1 or not roots[0].is_dir():
        raise ValueError("Expected one source distribution root")
    return roots[0]


def verify_source(archive, source, distribution):
    project = tomllib.loads((source / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    if project["name"] != "pmx-discrete-posterior" or project["version"] != VERSION:
        raise ValueError("Source examples must accompany pmx-discrete-posterior 0.2.0")
    if distribution == "wheel":
        # Wheel examples come from this checkout, so prove it matches the wheel's code.
        with zipfile.ZipFile(archive) as wheel:
            packaged = {name: wheel.read(name).replace(b"\r\n", b"\n") for name in wheel.namelist()
                        if name.startswith(PACKAGE + "/") and name.endswith(".py")}
        # Git may check out CRLF on the build runner and LF on the Linux test runner.
        source_files = {path.relative_to(source / "src").as_posix(): path.read_bytes().replace(b"\r\n", b"\n")
                        for path in (source / "src" / PACKAGE).rglob("*.py")}
        if not packaged or packaged != source_files:
            raise ValueError("Wheel does not match checkout source; use matching source examples")


def stage_tree(source, destination, folder):
    for path in sorted((source / folder).rglob("*")):
        relative = path.relative_to(source)
        if any(part in EXCLUDED or part.startswith("_tmp") or part.endswith("_output")
               for part in relative.parts):
            continue
        # Only frozen input data from this historical tree belongs in current demos.
        if relative.parts[:2] == ("examples", "manuscript_current"):
            if not (relative.as_posix() == "examples/manuscript_current/catalog.json"
                    or (relative.parts[2:3] == ("inputs",) and relative.suffix == ".json")):
                continue
        if path.is_symlink():
            raise ValueError(f"Cannot stage symlink: {path}")
        if path.is_file() and path.suffix.lower() in SUFFIXES:
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)


def stage_harness(source, destination, controls=ROOT):
    destination.mkdir()
    for folder in ("tests", "examples"):
        stage_tree(source, destination, folder)
    for relative in ("pyproject.toml", "tools/verify_reference.py"):
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, target)
    # CI controls are not shipped in the sdist; portable tests also exercise these.
    for relative in (".github/scripts/test_installed.py", ".github/scripts/test-installed.ps1",
                     ".github/workflows/ci.yml"):
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(controls / relative, target)
    if any(path.name in {"src", "vendor", PACKAGE} for path in destination.rglob("*")):
        raise ValueError("Harness must not contain production source or vendored packages")
    return {path.relative_to(destination).as_posix(): sha256(path)
            for path in sorted(destination.rglob("*")) if path.is_file()}


def run(command, cwd, env, log, stderr=None):
    print("Running: " + subprocess.list2cmdline([str(arg) for arg in command]), flush=True)
    with log.open("w", encoding="utf-8") as output:
        if stderr is None:
            result = subprocess.run(command, cwd=cwd, env=env, stdout=output, stderr=subprocess.STDOUT)
        else:
            with stderr.open("w", encoding="utf-8") as errors:
                result = subprocess.run(command, cwd=cwd, env=env, stdout=output, stderr=errors)
    print(log.read_text(encoding="utf-8"), end="", flush=True)
    if stderr is not None:
        print(stderr.read_text(encoding="utf-8"), end="", file=sys.stderr, flush=True)
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, command)


def check_commands(python, source, work, selection, reference):
    results = work / "results"
    return (
        [str(python), "-B", "-m", "pytest", "-c", str(source / "pyproject.toml"),
         "-o", "pythonpath=", "--strict-markers", "--require-installed", "--confcutdir", "tests",
         "-m", selection, "tests", "--basetemp", str(work / "pytest-tmp"),
         "-o", f"cache_dir={work / 'pytest-cache'}", "--junitxml", str(results / "pytest.xml")],
        [str(python), "-B", "tools/verify_reference.py", reference, "--require-installed"],
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distribution", required=True, choices=("wheel", "sdist"))
    parser.add_argument("--dependencies", required=True, choices=tuple(DEPENDENCIES))
    parser.add_argument("--selection", required=True, choices=("quick", "slow"))
    parser.add_argument("--reference", required=True, choices=("OXC", "TAC"))
    parser.add_argument("--archive", type=Path,
                        help="Existing wheel/sdist; default: the only matching file in .github/_ci/dist")
    parser.add_argument("--work-dir", type=Path,
                        help="Fresh output directory (must not exist); default: unique directory in .github/_ci")
    args = parser.parse_args(argv)
    ci = ROOT / ".github" / "_ci"
    archive = args.archive.resolve() if args.archive else select_archive(ci / "dist", args.distribution)
    suffix = ".whl" if args.distribution == "wheel" else ".tar.gz"
    if not archive.is_file() or not archive.name.endswith(suffix):
        parser.error(f"--archive must be an existing {args.distribution} ({suffix}): {archive}")
    # Every invocation gets a fresh environment; do not reuse or delete another run.
    if args.work_dir:
        work = args.work_dir.resolve()
        work.mkdir(parents=True, exist_ok=False)
    else:
        ci.mkdir(parents=True, exist_ok=True)
        work = Path(tempfile.mkdtemp(prefix=f"{args.distribution}-{args.dependencies}-",
                                     suffix=f"-{args.selection}", dir=ci))
    results = work / "results"
    results.mkdir()
    print(f"CI artifacts: {results}", flush=True)
    env = clean_environment(work)
    write_json(results / "distribution-sha256.json", {"Artifact": archive.name, "Hash": sha256(archive)})
    source = extract_source(archive, work / "extracted") if args.distribution == "sdist" else ROOT
    verify_source(archive, source, args.distribution)
    harness = work / "run"
    manifest = stage_harness(source, harness)
    write_json(results / "staged-inputs-sha256.json", {"source": str(source), "files": manifest})
    write_json(results / "selection.json", {**vars(args), "archive": str(archive), "work_dir": str(work)})
    run([sys.executable, "-B", "-m", "venv", str(work / "venv")], work, env, results / "venv.log")
    python = python_in(work / "venv")
    run([str(python), "-m", "pip", "install", str(archive), *DEPENDENCIES[args.dependencies],
         "pytest>=8,<10", "mpmath>=1.3,<2"], harness, env, results / "install.log")
    run([str(python), "-m", "pip", "check"], harness, env, results / "pip-check.txt")
    run([str(python), "-m", "pip", "freeze"], harness, env, results / "versions.txt")
    run([str(python), "-B", "-c", IMPORT_PROBE], harness, env, results / "installed-import.json")
    tests, reference = check_commands(python, harness, work, args.selection, args.reference)
    run(tests, harness, env, results / "pytest.log")
    run(reference, harness, env, results / f"{args.reference}-reference.json",
        stderr=results / f"{args.reference}-reference.stderr.log")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
