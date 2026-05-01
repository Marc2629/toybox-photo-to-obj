param(
    [ValidateSet("auto", "cpu", "cu124", "cu126", "cu128", "cu130")]
    [string]$TorchMode = "auto",

    [switch]$SkipHunyuan,
    [switch]$SkipTripoSR,
    [switch]$Force
)

$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Write-Ok {
    param([string]$Message)
    Write-Host "OK: $Message" -ForegroundColor Green
}

function Write-Warn {
    param([string]$Message)
    Write-Host "WARN: $Message" -ForegroundColor Yellow
}

function Require-Command {
    param(
        [string]$CommandName,
        [string]$InstallHint
    )

    if (-not (Get-Command $CommandName -ErrorAction SilentlyContinue)) {
        throw "$CommandName was not found. $InstallHint"
    }
}

function Get-NvidiaCudaVersion {
    if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
        return $null
    }

    $output = & nvidia-smi 2>$null
    if ($LASTEXITCODE -ne 0 -or -not $output) {
        return $null
    }

    $text = ($output | Out-String)
    $match = [regex]::Match($text, "CUDA Version:\s+([0-9]+(?:\.[0-9]+)?)")
    if (-not $match.Success) {
        return $null
    }

    return [version]$match.Groups[1].Value
}

function Resolve-TorchMode {
    param([string]$RequestedMode)

    if ($RequestedMode -ne "auto") {
        return $RequestedMode
    }

    $cudaVersion = Get-NvidiaCudaVersion
    if ($null -eq $cudaVersion) {
        Write-Warn "No NVIDIA CUDA driver was detected with nvidia-smi. Using CPU PyTorch."
        return "cpu"
    }

    Write-Ok "Detected NVIDIA driver with CUDA support $cudaVersion"

    if ($cudaVersion -ge [version]"13.0") {
        return "cu130"
    }

    if ($cudaVersion -ge [version]"12.8") {
        return "cu128"
    }

    if ($cudaVersion -ge [version]"12.6") {
        return "cu126"
    }

    if ($cudaVersion -ge [version]"12.4") {
        return "cu124"
    }

    Write-Warn "Your NVIDIA driver reports CUDA $cudaVersion, which is older than this script's CUDA wheel choices."
    Write-Warn "Install a newer NVIDIA driver for Hunyuan GPU mode, or rerun this script with -TorchMode cpu."
    return "cpu"
}

function Invoke-VenvPython {
    param([string[]]$Arguments)
    & ".\.venv\Scripts\python.exe" @Arguments
}

Write-Host "Toybox Photo to OBJ Windows setup" -ForegroundColor White
Write-Host "Project folder: $PWD"

Write-Step "Checking prerequisites"
Require-Command "py" "Install Python 3.11 from https://www.python.org/downloads/windows/ and keep the Python launcher enabled."
Require-Command "git" "Install Git from https://git-scm.com/download/win."

& py -3.11 --version
if ($LASTEXITCODE -ne 0) {
    throw "Python 3.11 was not found. Install it with: winget install -e --id Python.Python.3.11"
}
Write-Ok "Python 3.11 is available"

Write-Step "Creating or reusing .venv"
if ((Test-Path ".\.venv") -and $Force) {
    Write-Warn "Force was set, but this script does not delete .venv automatically."
    Write-Warn "Delete .venv yourself if you want a completely fresh install."
}

if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    & py -3.11 -m venv .venv
    Write-Ok "Created .venv"
} else {
    Write-Ok "Using existing .venv"
}

Invoke-VenvPython @("-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel")

Write-Step "Installing PyTorch"
$resolvedTorchMode = Resolve-TorchMode -RequestedMode $TorchMode
Write-Ok "Selected Torch mode: $resolvedTorchMode"

Invoke-VenvPython @("-m", "pip", "uninstall", "-y", "torch", "torchvision", "torchaudio")

switch ($resolvedTorchMode) {
    "cpu" {
        Invoke-VenvPython @("-m", "pip", "install", "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cpu")
    }
    "cu124" {
        Invoke-VenvPython @("-m", "pip", "install", "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu124")
    }
    "cu126" {
        Invoke-VenvPython @("-m", "pip", "install", "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu126")
    }
    "cu128" {
        Invoke-VenvPython @("-m", "pip", "install", "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu128")
    }
    "cu130" {
        Invoke-VenvPython @("-m", "pip", "install", "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu130")
    }
}

Invoke-VenvPython @("-c", "import torch; print('torch', torch.__version__); print('torch cuda build', torch.version.cuda); print('cuda available', torch.cuda.is_available()); print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')")

Write-Step "Installing base app dependencies"
Invoke-VenvPython @("-m", "pip", "install", "-r", "requirements.txt")
Invoke-VenvPython @("-m", "pip", "install", "numpy<2", "opencv-python==4.10.0.84")

if (-not $SkipHunyuan) {
    Write-Step "Installing Hunyuan3D-2.1 shape backend"
    New-Item -ItemType Directory -Force vendor | Out-Null

    if (-not (Test-Path ".\vendor\Hunyuan3D-2.1\.git")) {
        & git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git vendor\Hunyuan3D-2.1
    } else {
        Write-Ok "Hunyuan3D-2.1 already exists"
    }

    Invoke-VenvPython @("-m", "pip", "install", "-r", "vendor\Hunyuan3D-2.1\hy3dshape\requirements.txt")
    Invoke-VenvPython @("-m", "pip", "install", "timm", "torchdiffeq")
    Invoke-VenvPython @("-m", "pip", "install", "numpy<2", "opencv-python==4.10.0.84")
    Invoke-VenvPython @("-c", "import sys; sys.path.insert(0, r'vendor\Hunyuan3D-2.1\hy3dshape'); from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline; print('hunyuan shape import ok')")
}

if (-not $SkipTripoSR) {
    Write-Step "Installing optional TripoSR fallback backend"
    New-Item -ItemType Directory -Force vendor | Out-Null

    if (-not (Test-Path ".\vendor\TripoSR\.git")) {
        & git clone https://github.com/VAST-AI-Research/TripoSR.git vendor\TripoSR
    } else {
        Write-Ok "TripoSR already exists"
    }

    New-Item -ItemType Directory -Force temp | Out-Null
    Get-Content vendor\TripoSR\requirements.txt |
        Where-Object { $_ -notmatch '^Pillow==' } |
        Set-Content temp\triposr-requirements-windows.txt
    Invoke-VenvPython @("-m", "pip", "install", "-r", "temp\triposr-requirements-windows.txt")
}

Write-Step "Final app import check"
Invoke-VenvPython @("-c", "from app import parse_args, build_ui; build_ui(parse_args()); print('app ui build ok')")

Write-Host ""
Write-Host "Setup complete." -ForegroundColor Green
Write-Host "Run the app with:"
Write-Host "  .\run_app.ps1"
Write-Host ""
Write-Host "Manual launch:"
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "  python app.py"
