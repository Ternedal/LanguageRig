#requires -Version 7.3
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$Config,
    [ValidateRange(0, 2147483647)][int]$Gpu = 0,
    [string]$Distribution = 'Ubuntu-22.04',
    [string]$WslPython,
    [switch]$Execute,
    [switch]$FitProbe,
    [string]$Resume
)
$ErrorActionPreference = 'Stop'
$PSNativeCommandArgumentPassing = 'Standard'
$repoRoot = Split-Path $PSScriptRoot -Parent
if ($Resume -and -not $Execute) { throw '-Resume kraever -Execute.' }
if ($FitProbe -and $Execute) { throw '-FitProbe og -Execute kan ikke bruges samtidig.' }
if ($WslPython -and -not $WslPython.StartsWith('/')) { throw '-WslPython skal vaere en absolut Linux-sti.' }
$configPath = (Resolve-Path -LiteralPath $Config).ProviderPath
$wslExe = (Get-Command wsl.exe -ErrorAction Stop).Name
function Convert-WslPath {
    param([string]$Value)
    $converted = & $script:wslExe -d $script:Distribution --exec wslpath -a -u $Value
    if ($LASTEXITCODE -ne 0) { throw 'WSL-stikonvertering fejlede.' }
    $path = ($converted -join [char]10).Trim()
    if (-not $path.StartsWith('/') -or $path.Contains([char]10)) { throw 'Ugyldig sti fra wslpath.' }
    return $path
}
$repoWsl = Convert-WslPath $repoRoot
$configWsl = Convert-WslPath $configPath
$resumeWsl = if ($Resume) { Convert-WslPath (Resolve-Path -LiteralPath $Resume).ProviderPath } else { '-' }
$pythonWsl = if ($WslPython) { $WslPython } else { "$repoWsl/.venv-wsl/bin/python" }
$mode = if ($Execute) { 'execute' } elseif ($FitProbe) { 'fit' } else { 'check' }
& $wslExe -d $Distribution --exec bash -- "$repoWsl/scripts/train-pilot-wsl.sh" $configWsl $Gpu.ToString() $mode $pythonWsl $resumeWsl
if ($LASTEXITCODE -ne 0) { throw "WSL-piloten fejlede med kode $LASTEXITCODE. Se miljoerapporten eller fejlen ovenfor." }
