# Creates a Desktop shortcut to start the CCTV app via run_server.ps1
$ErrorActionPreference = 'Stop'

# Resolve paths
$projectRoot = Split-Path $PSScriptRoot -Parent
$runScript   = Join-Path $PSScriptRoot 'run_server.ps1'

if (-not (Test-Path $runScript)) {
    throw "Cannot find run_server.ps1 at $runScript"
}

$desktop = [Environment]::GetFolderPath('Desktop')
if (-not (Test-Path $desktop)) {
    throw "Desktop path not found: $desktop"
}

$shortcutName  = 'Start CCTV App.lnk'
$shortcutPath  = Join-Path $desktop $shortcutName
$targetPath    = "$env:WINDIR\System32\WindowsPowerShell\v1.0\powershell.exe"
$arguments     = "-NoProfile -ExecutionPolicy Bypass -File `"$runScript`""
$iconLocation  = "$env:WINDIR\System32\WindowsPowerShell\v1.0\powershell.exe,0"

$wsh = New-Object -ComObject WScript.Shell
$sc  = $wsh.CreateShortcut($shortcutPath)
$sc.TargetPath       = $targetPath
$sc.Arguments        = $arguments
$sc.WorkingDirectory = $projectRoot
$sc.WindowStyle      = 1
$sc.IconLocation     = $iconLocation
$sc.Description      = 'Start CCTV Face Recognition App'
$sc.Save()

Write-Host "Created shortcut:" $shortcutPath
