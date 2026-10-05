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
    foreach ($scriptName in @('prepare-pilot.ps1', 'train-pilot.ps1')) {
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
