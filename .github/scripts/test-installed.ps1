param(
    [Parameter(Mandatory)][ValidateSet("wheel", "sdist")][string]$Distribution,
    [Parameter(Mandatory)][ValidateSet("minimum", "current")][string]$Dependencies,
    [Parameter(Mandatory)][ValidateSet("quick", "slow")][string]$Selection,
    [Parameter(Mandatory)][ValidateSet("OXC", "TAC")][string]$Reference,
    [string]$Archive,
    [string]$WorkDir
)

$ErrorActionPreference = "Stop"
# Preserve the Windows entry point while sharing installed-package checks with Linux.
$Extra = @()
if ($Archive) { $Extra += @("--archive", $Archive) }
if ($WorkDir) { $Extra += @("--work-dir", $WorkDir) }
python -B (Join-Path $PSScriptRoot "test_installed.py") --distribution $Distribution --dependencies $Dependencies --selection $Selection --reference $Reference @Extra
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
