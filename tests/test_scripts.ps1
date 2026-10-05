#requires -Version 7.3
$ErrorActionPreference = 'Stop'
$PSNativeCommandArgumentPassing = 'Standard'
$repoRoot = Split-Path $PSScriptRoot -Parent
$testRoot = Join-Path ([IO.Path]::GetTempPath()) ([Guid]::NewGuid().ToString())
[IO.Directory]::CreateDirectory($testRoot) | Out-Null
function Assert-True {
    param([bool]$Condition, [string]$Message)
    if (-not $Condition) { throw $Message }
}
try {
    foreach ($scriptName in @('prepare-pilot.ps1', 'train-pilot.ps1', 'check-pilot-ready.ps1')) {
        $tokens = $null
        $parseErrors = $null
        [Management.Automation.Language.Parser]::ParseFile(
            (Join-Path $repoRoot "scripts/$scriptName"), [ref]$tokens, [ref]$parseErrors) | Out-Null
        Assert-True ($parseErrors.Count -eq 0) "PowerShell parse errors in $scriptName"
    }
    $bookPath = Join-Path $testRoot "Books with spaces ' literal $ ;"
    [IO.Directory]::CreateDirectory($bookPath) | Out-Null
    $fixturePath = Join-Path $testRoot 'fixture.py'
    $fixture = @'
import pathlib,sys
from zipfile import ZipFile
root=pathlib.Path(sys.argv[1])
for number in range(3):
    title=f"Vaerk {number}"
    with ZipFile(root/f"{number}.epub","w") as archive:
        archive.writestr("META-INF/container.xml",'<container><rootfiles><rootfile full-path="book.opf"/></rootfiles></container>')
        archive.writestr("book.opf",f'<package xmlns:dc="http://purl.org/dc/elements/1.1/"><metadata><dc:title>{title}</dc:title><dc:language>da</dc:language></metadata><manifest><item id="c" href="chapter.xhtml" media-type="application/xhtml+xml"/></manifest><spine><itemref idref="c"/></spine></package>')
        archive.writestr("chapter.xhtml",f"<h1>{title}</h1><p>"+(f"Indhold fra vaerk {number}. "*40)+"</p>")
'@
    [IO.File]::WriteAllText($fixturePath, $fixture, [Text.UTF8Encoding]::new($false))
    $pythonExe = (Get-Command python -ErrorAction Stop).Source
    & $pythonExe $fixturePath $bookPath
    Assert-True ($LASTEXITCODE -eq 0) 'Fixture generation failed'
    $workspaceRoot = Join-Path $testRoot 'Workspaces with spaces'
    $prepareOptions = @{Books=$bookPath; Name='smoke'; TrainingAllowed=$true; WorkspaceRoot=$workspaceRoot; Python=$pythonExe}
    & (Join-Path $repoRoot 'scripts/prepare-pilot.ps1') @prepareOptions
    $configPath = Join-Path $workspaceRoot 'smoke/configs/smoke.json'
    $receipt = Get-Content -LiteralPath (Join-Path $workspaceRoot 'smoke/pilots/smoke.json') -Raw | ConvertFrom-Json
    Assert-True ($receipt.status -eq 'prepared') 'PowerShell pilot was not prepared'
    Assert-True ($receipt.import.imported -eq 3) 'PowerShell import count mismatch'
    Assert-True (-not $receipt.training_executed) 'Preparation unexpectedly trained'
    Assert-True (Test-Path -LiteralPath $configPath) 'Prepared config missing'

    $global:wslCalls = [Collections.Generic.List[object]]::new()
    $global:stubBlock = $false
    $global:wslConfig = "/mnt/c/pilot with spaces ' literal " + '$' + ' ;.json'
    $global:wslResume = "/mnt/c/checkpoint with spaces ' literal " + '$' + ' ;20'
    function global:wsl.exe {
        $callArgs = @($args | ForEach-Object { [string]$_ })
        $global:wslCalls.Add($callArgs)
        $global:LASTEXITCODE = 0
        if ($callArgs[3] -eq 'wslpath') {
            if ($callArgs[-1].EndsWith('.json')) { return $global:wslConfig }
            if ($callArgs[-1].Contains('checkpoint')) { return $global:wslResume }
            return '/mnt/c/Language Rig'
        }
        if ($global:stubBlock) { $global:LASTEXITCODE = 7 }
    }
    & (Join-Path $repoRoot 'scripts/train-pilot.ps1') -Config $configPath -Gpu 1
    # PowerShell consumes the end-of-parameters marker for a function stub;
    # a native wsl.exe receives it. Compare the remaining data arguments.
    $last = @($global:wslCalls[-1] | Where-Object { $_ -ne '--' })
    Assert-True ($last.Count -eq 10) 'WSL argument boundaries changed'
    Assert-True ($last[5] -eq $global:wslConfig) 'WSL config argument was split or altered'
    Assert-True ($last[6] -eq '1' -and $last[7] -eq 'check') 'Default mode unexpectedly executed'
    Assert-True ($last[8] -eq '/mnt/c/Language Rig/.venv-wsl/bin/python') 'WSL Python path was altered'
    & (Join-Path $repoRoot 'scripts/train-pilot.ps1') -Config $configPath -Gpu 1 -FitProbe
    $last = @($global:wslCalls[-1] | Where-Object { $_ -ne '--' })
    Assert-True ($last[6] -eq '1' -and $last[7] -eq 'fit') 'Fit probe mode was altered'
    $checks = Join-Path $workspaceRoot 'smoke/checks'
    [IO.Directory]::CreateDirectory($checks) | Out-Null
    $doctorFixture = @{ training_environment_ready = $true; status = 'passed' } | ConvertTo-Json
    [IO.File]::WriteAllText((Join-Path $checks 'train-doctor.json'), $doctorFixture, [Text.UTF8Encoding]::new($false))
    $fitFixture = @{
        status = 'passed'
        training_gate = 'pass'
        gpu = 'NVIDIA GeForce RTX 3060'
        gpu_compute_capability = '8.6'
        gpu_total_memory_bytes = 12884901888
        model_id = 'fixture/model'
        resolved_revision = ('a' * 40)
        measured_sequence_tokens = 1024
        vram = @{
            minimum_observed_free_bytes = 2147483648
            minimum_observed_free_ratio = 0.1667
        }
    } | ConvertTo-Json -Depth 4
    [IO.File]::WriteAllText((Join-Path $checks 'fit-probe.json'), $fitFixture, [Text.UTF8Encoding]::new($false))
    $beforeReady = $global:wslCalls.Count
    $readyOutput = & (Join-Path $repoRoot 'scripts/check-pilot-ready.ps1') -Config $configPath -Gpu 1 | Out-String
    $readyCalls = @($global:wslCalls | Select-Object -Skip $beforeReady | Where-Object { $_[3] -ne 'wslpath' })
    Assert-True ($readyCalls.Count -eq 2) 'Readiness did not run exactly doctor then fit probe'
    $readyModes = @($readyCalls | ForEach-Object { @($_ | Where-Object { $_ -ne '--' })[7] })
    Assert-True ($readyModes[0] -eq 'check' -and $readyModes[1] -eq 'fit') 'Readiness gate order changed'
    Assert-True ($readyOutput.Contains('"status": "READY"')) 'Readiness summary was not READY'
    Assert-True ($readyOutput.Contains('NVIDIA GeForce RTX 3060')) 'Readiness summary lost GPU identity'
    $readinessReceipt = Join-Path $checks 'readiness.json'
    Assert-True (Test-Path -LiteralPath $readinessReceipt) 'Readiness receipt was not persisted'
    $readyReceipt = Get-Content -LiteralPath $readinessReceipt -Raw | ConvertFrom-Json
    Assert-True ($readyReceipt.format -eq 'languagerig-readiness/v1') 'Readiness receipt format mismatch'
    Assert-True ($readyReceipt.status -eq 'READY') 'Readiness receipt status mismatch'
    Assert-True ($readyReceipt.doctor_sha256.Length -eq 64) 'Doctor receipt hash missing'
    Assert-True ($readyReceipt.fit_sha256.Length -eq 64) 'Fit receipt hash missing'
    $exclusiveFailed = $false
    try { & (Join-Path $repoRoot 'scripts/train-pilot.ps1') -Config $configPath -FitProbe -Execute }
    catch { $exclusiveFailed = $true }
    Assert-True $exclusiveFailed '-FitProbe and -Execute were allowed together'
    $checkpoint = Join-Path $testRoot 'checkpoint with spaces'
    [IO.Directory]::CreateDirectory($checkpoint) | Out-Null
    & (Join-Path $repoRoot 'scripts/train-pilot.ps1') -Config $configPath -Execute -Resume $checkpoint
    $last = @($global:wslCalls[-1] | Where-Object { $_ -ne '--' })
    Assert-True ($last[7] -eq 'execute' -and $last[9] -eq $global:wslResume) 'Resume arguments were altered'
    $global:stubBlock = $true
    $failed = $false
    try { & (Join-Path $repoRoot 'scripts/train-pilot.ps1') -Config $configPath -Execute }
    catch { $failed = $true }
    Assert-True $failed 'WSL failure was ignored'
    Write-Host 'PowerShell preparation and WSL transport smoke passed (WSL is a controlled stub).'
} finally {
    Remove-Item Function:\wsl.exe -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $testRoot -Recurse -Force
}
exit 0
