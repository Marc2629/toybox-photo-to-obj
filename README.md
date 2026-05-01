# Toybox Photo to OBJ

This is a local Windows app that turns one photo of an object into an OBJ mesh for Toybox import.

It uses:

- Gradio for the local web UI
- rembg for background removal
- Hunyuan3D-2.1 by default for image-to-3D generation
- TripoSR as an optional fallback backend
- trimesh for light mesh cleanup and OBJ/STL export

The app does not upload your photo or mesh. The first run downloads model files; after that, generation uses the local cache. Toybox still handles downstream Toybox-specific post-processing.

## What You Need

- Windows 11
- PowerShell
- Git
- Python 3.11
- An NVIDIA GPU is strongly recommended for Hunyuan3D-2.1

Python 3.14 is too new for some of these 3D/AI packages. Install Python 3.11 even if you already have another Python version.

## Step 1: Open PowerShell

Open PowerShell and go to this project folder:

```powershell
cd "C:\Users\Marc\Documents\New project"
```

## Easy Install

For most people, use the setup script:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.\setup_windows.ps1
```

The setup script will:

- check for Python 3.11
- create `.venv`
- detect your NVIDIA driver with `nvidia-smi`
- choose a CUDA PyTorch wheel when possible
- install the app dependencies
- clone/install Hunyuan3D-2.1 shape generation
- clone/install TripoSR as a fallback
- run import checks

After setup finishes, launch the app:

```powershell
.\run_app.ps1
```

Then open:

```text
http://127.0.0.1:7860
```

Manual setup instructions are below if you want to see or control each step.

### Setup Script Options

Force CPU PyTorch:

```powershell
.\setup_windows.ps1 -TorchMode cpu
```

Force a specific CUDA PyTorch wheel:

```powershell
.\setup_windows.ps1 -TorchMode cu128
```

If your NVIDIA driver reports CUDA 13.0, you can also use:

```powershell
.\setup_windows.ps1 -TorchMode cu130
```

Skip TripoSR fallback:

```powershell
.\setup_windows.ps1 -SkipTripoSR
```

Skip Hunyuan3D:

```powershell
.\setup_windows.ps1 -SkipHunyuan
```

### Run Script Options

Run Hunyuan on GPU:

```powershell
.\run_app.ps1 -Backend hunyuan3d -Device cuda:0
```

Run TripoSR:

```powershell
.\run_app.ps1 -Backend triposr
```

Use a different port:

```powershell
.\run_app.ps1 -Port 7861
```

Start at lower resolution:

```powershell
.\run_app.ps1 -MeshResolution 256
```

## Step 2: Install Python 3.11

Check whether Python 3.11 is already installed:

```powershell
py -3.11 --version
```

If that prints `Python 3.11.x`, continue to Step 3.

If it says Python 3.11 is not found, install it:

```powershell
winget install -e --id Python.Python.3.11
```

Close PowerShell, reopen it, go back to the project folder, and check again:

```powershell
cd "C:\Users\Marc\Documents\New project"
py -3.11 --version
```

Manual option: download Python 3.11 from `https://www.python.org/downloads/windows/`. During install, keep the Python launcher enabled.

## Step 3: Create The Virtual Environment

Run these commands:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python --version
```

The last command should print `Python 3.11.x`.

Any time you open a new PowerShell window later, activate the environment again:

```powershell
cd "C:\Users\Marc\Documents\New project"
.\.venv\Scripts\Activate.ps1
```

You should see `(.venv)` at the start of your prompt.

## Step 4: Install PyTorch For GPU

If you have an NVIDIA GPU, install CUDA PyTorch:

```powershell
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

Check that PyTorch can see your GPU:

```powershell
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"
```

You want to see `True` and your NVIDIA GPU name.

If you do not have an NVIDIA GPU, skip this step and use TripoSR later with:

```powershell
python app.py --backend triposr --device cpu
```

Hunyuan3D-2.1 is intended for GPU use and needs roughly 10 GB VRAM.

## Step 5: Install This App

Install the base app dependencies:

```powershell
pip install -r requirements.txt
```

## Step 6: Install Hunyuan3D-2.1

Hunyuan3D-2.1 is the default backend and usually gives better geometry than TripoSR.

Clone the Hunyuan3D repo:

```powershell
New-Item -ItemType Directory -Force vendor
git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git vendor\Hunyuan3D-2.1
```

Install only the shape-generation requirements:

```powershell
pip install -r vendor\Hunyuan3D-2.1\hy3dshape\requirements.txt
pip install timm torchdiffeq
pip install "numpy<2" "opencv-python==4.10.0.84"
```

Do not install `vendor\Hunyuan3D-2.1\requirements.txt` for this MVP. That top-level file includes Blender/PBR texture packages such as `bpy`, which are not needed here and can fail on Windows.

Check that Hunyuan imports:

```powershell
python -c "import sys; sys.path.insert(0, r'vendor\Hunyuan3D-2.1\hy3dshape'); from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline; print('hunyuan shape import ok')"
```

## Step 7: Run The App

Start the local web app:

```powershell
python app.py
```

Wait until you see:

```text
Running on local URL: http://127.0.0.1:7860
```

Open this address in your browser:

```text
http://127.0.0.1:7860
```

## Step 8: Use The App

1. Upload a photo with one clear object.
2. Leave backend set to `Hunyuan3D-2.1`.
3. Start with these settings:
   - Mesh Resolution: `384`
   - Hunyuan Steps: `30`
   - Hunyuan Guidance: `5`
4. Click **Generate OBJ**.
5. Wait for generation to finish.
6. Download the OBJ file.

Output files are saved in:

```text
outputs\
```

Temporary processed images and preview files are saved in:

```text
temp\
```

If the mesh looks wrong, check:

```text
temp\processed.png
```

That is the image being sent into the 3D model.

## Recommended Photos

Best results come from:

- One object only
- Plain background
- Even lighting
- Object fills most of the image
- 3/4 angle view
- No transparent, shiny, fuzzy, or very thin parts
- Clean object edges after background removal

Single-image 3D generation guesses the hidden side of the object, so exact replicas are not guaranteed.

## Useful Run Commands

Run with Hunyuan3D-2.1 on GPU:

```powershell
python app.py --backend hunyuan3d --device cuda:0
```

Run with TripoSR instead:

```powershell
$env:PYTHONPATH = "$PWD\vendor\TripoSR;$env:PYTHONPATH"
python app.py --backend triposr
```

Use a different port:

```powershell
python app.py --port 7861
```

Use lower default resolution:

```powershell
python app.py --mesh-resolution 256
```

Use higher default resolution:

```powershell
python app.py --mesh-resolution 512
```

## Optional: Install TripoSR Fallback

TripoSR is faster and lighter than Hunyuan3D-2.1, but mesh quality is usually lower.

Clone TripoSR:

```powershell
New-Item -ItemType Directory -Force vendor
git clone https://github.com/VAST-AI-Research/TripoSR.git vendor\TripoSR
```

Install TripoSR requirements, skipping its old Pillow pin:

```powershell
Get-Content vendor\TripoSR\requirements.txt | Where-Object { $_ -notmatch '^Pillow==' } | Set-Content temp\triposr-requirements-windows.txt
pip install -r temp\triposr-requirements-windows.txt
```

Run with TripoSR:

```powershell
$env:PYTHONPATH = "$PWD\vendor\TripoSR;$env:PYTHONPATH"
python app.py --backend triposr
```

## What The Settings Mean

Mesh Backend:

- `Hunyuan3D-2.1`: better quality, needs a good NVIDIA GPU
- `TripoSR`: faster/lighter fallback

Mesh Resolution:

- Higher values create denser meshes and take more memory/time.
- Try `256`, `384`, or `512`.

Hunyuan Steps:

- More steps can improve quality but take longer.
- Start with `30`.

Hunyuan Guidance:

- Controls how strongly the model follows the input image.
- Start with `5`.
- Try `7` if the result is too generic.
- Try `3-4` if the result is noisy or distorted.

## Troubleshooting

If PowerShell says scripts are disabled when activating `.venv`:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.\.venv\Scripts\Activate.ps1
```

If `python --version` prints Python 3.14:

```powershell
Deactivate
Remove-Item -Recurse -Force .venv
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version
```

If PyTorch does not see your GPU:

```powershell
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.cuda.is_available())"
```

If you get a NumPy/trimesh error about `ptp`:

```powershell
pip install "numpy<2" "opencv-python==4.10.0.84"
```

If Hunyuan says `No module named timm`:

```powershell
pip install timm torchdiffeq
```

If Hunyuan runs out of GPU memory:

- Lower Mesh Resolution to `256`
- Lower Hunyuan Steps to `20`
- Close other GPU-heavy apps
- Try TripoSR instead

If the preview is black:

- Regenerate with the latest code.
- The app now strips black material colors from preview/export meshes.

## Project Layout

```text
app.py
setup_windows.ps1
run_app.ps1
requirements.txt
README.md
src/
  background.py
  preprocess.py
  inference.py
  mesh_utils.py
  export.py
outputs/
temp/
vendor/
```

## Important Scope Notes

This MVP does not include:

- slicing
- printability analysis
- authentication
- Toybox cloud upload
- mobile support

It creates OBJ/STL files locally. Toybox handles downstream Toybox-specific work.
