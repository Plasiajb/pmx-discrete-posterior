param(
    [Parameter(Mandatory)][ValidateSet("wheel", "sdist")][string]$Distribution,
    [Parameter(Mandatory)][ValidateSet("minimum", "current")][string]$Dependencies,
    [Parameter(Mandatory)][ValidateSet("quick", "slow")][string]$Selection,
    [Parameter(Mandatory)][ValidateSet("OXC", "TAC")][string]$Reference
)

$ErrorActionPreference = "Stop"
$Repository = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
$Work = Join-Path $Repository ".github/_ci/$Distribution-$Selection"
$Harness = Join-Path $Work "run"
$Results = Join-Path $Work "results"
New-Item -ItemType Directory -Force -Path $Work, $Harness, $Results | Out-Null
$env:PYTHONPATH = ""
$env:PYTHONDONTWRITEBYTECODE = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PIP_CACHE_DIR = Join-Path $Work "pip-cache"
$env:TEMP = Join-Path $Work "tmp"
$env:TMP = $env:TEMP
New-Item -ItemType Directory -Force -Path $env:TEMP | Out-Null

$Pattern = if ($Distribution -eq "wheel") { "*.whl" } else { "*.tar.gz" }
$Archives = @(Get-ChildItem -LiteralPath (Join-Path $Repository ".github/_ci/dist") -Filter $Pattern -File)
if ($Archives.Count -ne 1) { throw "Expected exactly one $Distribution distribution" }
$Archive = $Archives[0].FullName
$Source = $Repository
if ($Distribution -eq "sdist") {
    $Extracted = Join-Path $Work "extracted"
    New-Item -ItemType Directory -Force -Path $Extracted | Out-Null
    python -c "import sys,tarfile; tarfile.open(sys.argv[1]).extractall(sys.argv[2], filter='data')" $Archive $Extracted
    if ($LASTEXITCODE -ne 0) { throw "sdist extraction failed" }
    $Roots = @(Get-ChildItem -LiteralPath $Extracted -Directory)
    if ($Roots.Count -ne 1) { throw "Expected one source distribution root" }
    $Source = $Roots[0].FullName
}

# No production src tree is copied into the test harness.
foreach ($Folder in @("tests", "tests/fixtures", "examples", "tools")) {
    New-Item -ItemType Directory -Force -Path (Join-Path $Harness $Folder) | Out-Null
}
Get-ChildItem -LiteralPath (Join-Path $Source "tests") -Filter "*.py" -File | ForEach-Object {
    Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $Harness "tests")
}
Get-ChildItem -LiteralPath (Join-Path $Source "tests/fixtures") -Recurse -File |
    Where-Object { $_.Extension -in @(".py", ".json", ".md") } | ForEach-Object {
        $Relative = [System.IO.Path]::GetRelativePath($Source, $_.FullName)
        $Destination = Join-Path $Harness $Relative
        New-Item -ItemType Directory -Force -Path (Split-Path $Destination) | Out-Null
        Copy-Item -LiteralPath $_.FullName -Destination $Destination
    }
foreach ($File in @("examples/reference_models.py", "examples/posterior_reference.py", "tools/verify_reference.py")) {
    Copy-Item -LiteralPath (Join-Path $Source $File) -Destination (Join-Path $Harness $File)
}
if (Test-Path -LiteralPath (Join-Path $Harness "src")) { throw "Harness must not contain src" }
python -m venv (Join-Path $Work "venv")
if ($LASTEXITCODE -ne 0) { throw "venv creation failed" }
$Python = Join-Path $Work "venv/Scripts/python.exe"
$Numeric = if ($Dependencies -eq "minimum") {
    @("numpy==1.26.0", "scipy==1.16.2")
} else {
    @("numpy>=2,<3", "scipy>=1.16.2,<2")
}
& $Python -m pip install $Archive @Numeric "pytest>=8,<10" "mpmath>=1.3,<2"
if ($LASTEXITCODE -ne 0) { throw "Clean distribution installation failed" }
& $Python -m pip check
if ($LASTEXITCODE -ne 0) { throw "Installed dependencies are inconsistent" }
& $Python -m pip freeze | Tee-Object -FilePath (Join-Path $Results "versions.txt")
if ($LASTEXITCODE -ne 0) { throw "Could not record dependency versions" }
Get-FileHash -LiteralPath $Archive -Algorithm SHA256 |
    Select-Object Hash, @{Name="Artifact";Expression={Split-Path $_.Path -Leaf}} |
    ConvertTo-Json | Set-Content -LiteralPath (Join-Path $Results "distribution-sha256.json") -Encoding utf8
$Cache = Join-Path $Work "pytest-cache"

Push-Location $Harness
try {
    & $Python -m pytest -c (Join-Path $Repository "pyproject.toml") -o pythonpath= --strict-markers --require-installed --confcutdir tests -m $Selection tests --basetemp (Join-Path $Work "pytest-tmp") -o "cache_dir=$Cache" --junitxml (Join-Path $Results "pytest.xml")
    if ($LASTEXITCODE -ne 0) { throw "Installed $Selection suite failed" }
    & $Python tools/verify_reference.py $Reference --require-installed |
        Tee-Object -FilePath (Join-Path $Results "$Reference-reference.json")
    if ($LASTEXITCODE -ne 0) { throw "$Reference reference verification failed" }
} finally {
    Pop-Location
}
