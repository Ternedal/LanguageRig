#requires -Version 7.3
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$Books,
    [Parameter(Mandatory)][ValidatePattern('^[A-Za-z0-9][A-Za-z0-9._-]{0,60}$')][string]$Name,
    [switch]$TrainingAllowed,
    [string]$Language,
    [ValidateSet('unknown', 'fiction', 'nonfiction')][string]$Genre = 'unknown',
    [string[]]$Topic = @(),
    [string]$WorkspaceRoot = 'data/pilots',
    [string]$Python,
    [ValidateRange(1, 2147483647)][int]$MaxSteps = 100
)
$ErrorActionPreference = 'Stop'
$PSNativeCommandArgumentPassing = 'Standard'
$env:PYTHONIOENCODING = 'utf-8'
$repoRoot = Split-Path $PSScriptRoot -Parent
if (-not $TrainingAllowed) { throw 'Angiv -TrainingAllowed for det materiale, du har valgt til traening.' }
$bookPath = (Resolve-Path -LiteralPath $Books).ProviderPath
$workspacePath = if ([IO.Path]::IsPathRooted($WorkspaceRoot)) {
    Join-Path $WorkspaceRoot $Name
} else { Join-Path (Join-Path $repoRoot $WorkspaceRoot) $Name }
function Invoke-CheckedPython {
    param([string[]]$Arguments)
    & $script:pythonExe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "LanguageRig-kommando fejlede med kode $LASTEXITCODE. Se rapporten eller fejlen ovenfor." }
}
if ($Python) {
    $pythonExe = (Get-Command $Python -ErrorAction Stop).Source
} else {
    $pythonExe = Join-Path $repoRoot '.venv/Scripts/python.exe'
    if (-not (Test-Path -LiteralPath $pythonExe)) {
        $launcher = Get-Command py -ErrorAction SilentlyContinue
        if ($launcher) {
            & $launcher.Source -3.12 -m venv (Join-Path $repoRoot '.venv')
        } else {
            $launcher = Get-Command python -ErrorAction Stop
            & $launcher.Source -m venv (Join-Path $repoRoot '.venv')
        }
        if ($LASTEXITCODE -ne 0) { throw 'Oprettelse af Python-miljoe fejlede.' }
    }
}
Invoke-CheckedPython -Arguments @('-m', 'pip', 'install', '-e', $repoRoot)
$cliArgs = @('-m', 'languagerig', '--workspace', $workspacePath, 'prepare-pilot',
             $bookPath, '--name', $Name, '--training-allowed', '--genre', $Genre,
             '--max-steps', $MaxSteps.ToString())
if ($Language) { $cliArgs += @('--language', $Language) }
foreach ($entry in $Topic) { $cliArgs += @('--topic', $entry) }
Invoke-CheckedPython -Arguments $cliArgs
Invoke-CheckedPython -Arguments @('-m', 'languagerig', '--workspace', $workspacePath, 'doctor',
                       '--config', (Join-Path $workspacePath "configs/$Name.json"),
                       '--report', (Join-Path $workspacePath 'checks/prepare-doctor.json'))
Write-Host "Pilot klar. Konfiguration: $(Join-Path $workspacePath "configs/$Name.json")"
