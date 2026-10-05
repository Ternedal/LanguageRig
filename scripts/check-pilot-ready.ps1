#requires -Version 7.3
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$Config,
    [ValidateRange(0, 2147483647)][int]$Gpu = 0,
    [string]$Distribution = 'Ubuntu-22.04',
    [string]$WslPython
)
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path $PSScriptRoot -Parent
$trainScript = Join-Path $PSScriptRoot 'train-pilot.ps1'
$configPath = (Resolve-Path -LiteralPath $Config).ProviderPath
$configData = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
if (-not $configData.output_dir) { throw 'Konfigurationen mangler output_dir.' }

$outputPath = [IO.Path]::GetFullPath((Join-Path (Split-Path $configPath -Parent) ([string]$configData.output_dir)))
$workspacePath = Split-Path (Split-Path $outputPath -Parent) -Parent
$fitReportPath = Join-Path $workspacePath 'checks/fit-probe.json'
$doctorReportPath = Join-Path $workspacePath 'checks/train-doctor.json'

$common = @{Config=$configPath; Gpu=$Gpu; Distribution=$Distribution}
if ($WslPython) { $common.WslPython = $WslPython }

Write-Host '=== LanguageRig readiness 1/2: environment ==='
& $trainScript @common

Write-Host '=== LanguageRig readiness 2/2: model fit ==='
& $trainScript @common -FitProbe

if (-not (Test-Path -LiteralPath $doctorReportPath)) {
    throw "Readiness failed: doctor report mangler: $doctorReportPath"
}
if (-not (Test-Path -LiteralPath $fitReportPath)) {
    throw "Readiness failed: fit-probe report mangler: $fitReportPath"
}

$doctor = Get-Content -LiteralPath $doctorReportPath -Raw | ConvertFrom-Json
$fit = Get-Content -LiteralPath $fitReportPath -Raw | ConvertFrom-Json
if (-not $doctor.training_environment_ready) {
    throw 'Readiness failed: training environment er ikke godkendt.'
}
if ($fit.status -ne 'passed' -or $fit.training_gate -ne 'pass') {
    throw "Readiness failed: fit-probe gate er $($fit.training_gate)."
}

$totalGiB = [Math]::Round(([double]$fit.gpu_total_memory_bytes / 1GB), 2)
$freeGiB = [Math]::Round(([double]$fit.vram.minimum_observed_free_bytes / 1GB), 2)
$freePct = [Math]::Round(([double]$fit.vram.minimum_observed_free_ratio * 100), 1)
$summary = [ordered]@{
    status = 'READY'
    config = $configPath
    workspace = $workspacePath
    gpu = $fit.gpu
    compute_capability = $fit.gpu_compute_capability
    total_vram_gib = $totalGiB
    minimum_free_vram_gib = $freeGiB
    minimum_free_vram_percent = $freePct
    model_id = $fit.model_id
    resolved_revision = $fit.resolved_revision
    sequence_tokens = $fit.measured_sequence_tokens
    doctor_report = $doctorReportPath
    fit_report = $fitReportPath
    next_command = ".\\scripts\\train-pilot.ps1 -Config '$configPath' -Gpu $Gpu -Execute"
}

Write-Host ''
Write-Host '=== LANGUAGERIG READY ==='
Write-Host ("GPU: {0} | CC {1} | VRAM {2} GiB" -f $summary.gpu, $summary.compute_capability, $summary.total_vram_gib)
Write-Host ("Minimum observed free VRAM: {0} GiB ({1}%)" -f $summary.minimum_free_vram_gib, $summary.minimum_free_vram_percent)
Write-Host ("Model: {0} @ {1}" -f $summary.model_id, $summary.resolved_revision)
Write-Host ("Sequence: {0} tokens" -f $summary.sequence_tokens)
Write-Host ''
$summary | ConvertTo-Json -Depth 4
