param(
    [switch] $NoBrowser,
    [switch] $Https,
    [string] $Hosts = "localhost,127.0.0.1"
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
# Project root is the parent of scripts folder
$proj = Split-Path -Parent $root
Push-Location $proj
try {
    $venvPy = Join-Path $proj ".venv/Scripts/python.exe"
    if (!(Test-Path $venvPy)) {
        Write-Error "Venv python not found at $venvPy. Activate venv or install dependencies."
        exit 1
    }
    if ($Https) {
        $outDir = Join-Path $proj "models/certs"
        if (!(Test-Path (Join-Path $outDir 'cert.pem')) -or !(Test-Path (Join-Path $outDir 'key.pem'))) {
            & $venvPy (Join-Path $proj 'scripts/generate_dev_cert.py') --hosts $Hosts --out $outDir | Write-Host
        }
        $env:SSL_CERT = Join-Path $outDir 'cert.pem'
        $env:SSL_KEY  = Join-Path $outDir 'key.pem'
    }
    # Start server in a new window so this script can return
    Start-Process -FilePath $venvPy -ArgumentList "app.py"
    if (-not $NoBrowser) {
        Start-Sleep -Seconds 1
        if ($Https) { Start-Process "https://127.0.0.1:5000/" } else { Start-Process "http://127.0.0.1:5000/" }
    }
}
finally {
    Pop-Location
}
