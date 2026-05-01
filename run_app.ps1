param(
    [ValidateSet("hunyuan3d", "triposr")]
    [string]$Backend = "hunyuan3d",

    [string]$Device = "auto",
    [int]$Port = 7860,
    [int]$MeshResolution = 384
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    throw ".venv was not found. Run .\setup_windows.ps1 first."
}

if ($Backend -eq "triposr") {
    $tripoPath = Join-Path $PWD "vendor\TripoSR"
    if (-not (Test-Path $tripoPath)) {
        throw "TripoSR was not found at vendor\TripoSR. Run .\setup_windows.ps1 or launch with -Backend hunyuan3d."
    }
    $env:PYTHONPATH = "$tripoPath;$env:PYTHONPATH"
}

Write-Host "Starting Toybox Photo to OBJ..." -ForegroundColor Cyan
Write-Host "Backend: $Backend"
Write-Host "Device: $Device"
Write-Host "URL: http://127.0.0.1:$Port"
Write-Host ""

& ".\.venv\Scripts\python.exe" app.py --backend $Backend --device $Device --port $Port --mesh-resolution $MeshResolution
