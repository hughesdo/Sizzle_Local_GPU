@echo off
REM ===========================================================================
REM  SIZZLE - first-time setup for Windows + NVIDIA. Double-click me once.
REM ===========================================================================
REM  Builds a local Python 3.12 environment with CUDA PyTorch, installs
REM  Lightricks' LTX-2 inference packages, and creates your .env.
REM  Safe to re-run: it reuses an existing .venv and won't overwrite your .env.
REM
REM  It does NOT download the ~128 GB of model weights - that is a separate,
REM  resumable step it offers you at the end (see models\README.md).
REM
REM  ---------------------------------------------------------------------------
REM  GPU PORTABILITY - the one thing you may need to change
REM  ---------------------------------------------------------------------------
REM  Step 6 installs torch from PyTorch's CUDA index, NOT from PyPI (PyPI's
REM  default torch is CPU-only and would silently give you a box that cannot
REM  render). It defaults to cu132 because Blackwell (sm_120) requires it;
REM  cu132 also covers every older CUDA card, so this is usually right.
REM
REM  If nvidia-smi reports a CUDA version below 13, override without editing:
REM      set SIZZLE_TORCH_INDEX=https://download.pytorch.org/whl/cu128
REM      set SIZZLE_TORCH_SPEC=torch
REM      install.bat
REM
REM  See SETUP.md section 3.1 for the driver -^> wheel table.
REM
REM  On Linux/macOS (i.e. most rented GPU boxes) use install.sh instead.
REM ===========================================================================
setlocal enabledelayedexpansion
cd /d "%~dp0"

REM --- GPU-portability knobs (override from the environment) ---------------
if not defined SIZZLE_TORCH_INDEX set "SIZZLE_TORCH_INDEX=https://download.pytorch.org/whl/cu132"
if not defined SIZZLE_TORCH_SPEC  set "SIZZLE_TORCH_SPEC=torch==2.13.0+cu132"
REM torchaudio for cu132 currently only exists on the test index. If you moved
REM SIZZLE_TORCH_INDEX to a stable cuXXX channel, match it here too.
if not defined SIZZLE_TORCHAUDIO_INDEX set "SIZZLE_TORCHAUDIO_INDEX=https://download.pytorch.org/whl/test/cu132/"

echo.
echo  =========================================================
echo   SIZZLE local-GPU setup
echo  =========================================================
echo.

REM --- 1. uv (installs and manages Python for us) --------------------------
echo ^>^> Checking for uv...
where uv >nul 2>nul
if errorlevel 1 (
  echo    uv not found - installing it with winget...
  winget install --id astral-sh.uv --accept-source-agreements --accept-package-agreements
  where uv >nul 2>nul
  if errorlevel 1 (
    echo.
    echo !! uv still not on PATH. Close this window, open a NEW one, and re-run
    echo    install.bat ^(winget updates PATH only for new shells^).
    echo    Or install manually: https://docs.astral.sh/uv/getting-started/installation/
    pause
    exit /b 1
  )
)

REM --- 2. ffmpeg ------------------------------------------------------------
echo ^>^> Checking for ffmpeg...
where ffmpeg >nul 2>nul
if errorlevel 1 (
  echo    ffmpeg not found - installing it with winget...
  winget install --id Gyan.FFmpeg --accept-source-agreements --accept-package-agreements
  echo    ^(if ffmpeg is still missing later, open a NEW terminal so PATH refreshes^)
)

REM --- 3. NVIDIA driver sanity check ---------------------------------------
echo ^>^> Checking for an NVIDIA GPU...
where nvidia-smi >nul 2>nul
if errorlevel 1 (
  echo.
  echo !! nvidia-smi not found. Sizzle renders on an NVIDIA GPU via CUDA.
  echo    Install a current NVIDIA driver, or run with SIZZLE_BACKEND=mock
  echo    to explore the app without a model.
  echo.
  pause
) else (
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
)

REM --- 4. the LTX-2 inference packages -------------------------------------
REM  Sizzle drives Lightricks' own pipeline code. We vendor it as a sibling
REM  checkout and pip-install it editable, so upgrading is just a git pull.
if not exist "LTX-2\packages\ltx-pipelines\pyproject.toml" (
  echo ^>^> Cloning Lightricks/LTX-2 ^(inference pipelines^)...
  where git >nul 2>nul
  if errorlevel 1 (
    echo !! git not found. Install Git for Windows, then re-run install.bat.
    echo    https://git-scm.com/download/win
    pause
    exit /b 1
  )
  git clone --depth 1 https://github.com/Lightricks/LTX-2.git LTX-2
  if errorlevel 1 (
    echo !! Failed to clone LTX-2.
    pause
    exit /b 1
  )
) else (
  echo ^>^> LTX-2 checkout already present
)

REM --- 5. Python 3.12 + venv ------------------------------------------------
echo ^>^> Installing Python 3.12 ^(via uv^)...
uv python install 3.12

if not exist ".venv\Scripts\python.exe" (
  echo ^>^> Creating virtual environment ^(.venv^)...
  uv venv --python 3.12 .venv
) else (
  echo ^>^> Reusing existing .venv
)
set "PY=.venv\Scripts\python.exe"

REM --- 6. CUDA PyTorch ------------------------------------------------------
REM  cu132 wheels cover Blackwell (sm_120) as well as older cards. This is a
REM  ~3 GB download the first time. Retarget with SIZZLE_TORCH_INDEX / _SPEC
REM  (see the header and SETUP.md section 3.1) if this host's driver is older.
echo.
echo ^>^> Installing PyTorch from %SIZZLE_TORCH_INDEX% ^(large download, be patient^)...
uv pip install --python "%PY%" --index-url "%SIZZLE_TORCH_INDEX%" "%SIZZLE_TORCH_SPEC%"
if errorlevel 1 goto :piperr
uv pip install --python "%PY%" --index-url "%SIZZLE_TORCHAUDIO_INDEX%" torchaudio
if errorlevel 1 goto :piperr

REM --- 7. LTX packages + Sizzle's own deps ---------------------------------
REM  Only ltx-core and ltx-pipelines. NOT ltx-kernels (optional fused CUDA
REM  kernels, builds from source) and NOT ltx-trainer (training only).
REM  Upstream API drift here is the likeliest rebuild failure - if the verify
REM  step below cannot import the pipeline, check out the pinned commit in
REM  VENDOR-PINS.txt and re-run.
echo ^>^> Installing LTX-2 inference packages...
uv pip install --python "%PY%" -e LTX-2\packages\ltx-core -e LTX-2\packages\ltx-pipelines
if errorlevel 1 goto :piperr

echo ^>^> Installing Sizzle dependencies...
uv pip install --python "%PY%" -r requirements.txt
if errorlevel 1 goto :piperr

REM --- 8. .env --------------------------------------------------------------
REM  In the backup repo .env is committed, so this is normally a no-op - which
REM  is intended. Secrets belong in .env.local (gitignored), NOT here.
REM  See SETUP.md section 7.
if not exist ".env" (
  echo ^>^> Creating .env from .env.example
  copy /y ".env.example" ".env" >nul
) else (
  echo ^>^> .env already exists - leaving it as-is
)

REM --- 9. verify ------------------------------------------------------------
echo.
echo ^>^> Verifying the install...
"%PY%" -c "import torch; print('   torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
"%PY%" -c "from ltx_pipelines.a2vid_two_stage import A2VidPipelineTwoStage; print('   LTX-2 audio-to-video pipeline: OK')"
if errorlevel 1 goto :piperr

echo.
echo  =========================================================
echo   Setup complete.
echo  =========================================================
echo.
echo   Still needed: the model weights ^(~128 GB, resumable^) - models\README.md
echo.
set /p DLNOW="  Download them now? [y/N] "
if /i "!DLNOW!"=="y" (
  "%PY%" scripts\download_models.py
) else (
  echo.
  echo   Run this when you're ready:
  echo       .venv\Scripts\python.exe scripts\download_models.py
)

echo.
echo   Then:
echo     startup.bat   run it locally at http://127.0.0.1:8000
echo     host.bat      share it with friends over a Cloudflare tunnel
echo.
echo   NOTE: the Gemma text encoder is a GATED download. Accept the license at
echo     https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized
echo   then run:  .venv\Scripts\hf.exe auth login
echo   Full details: models\README.md
echo.
echo   Rebuilding on different hardware? Read SETUP.md section 4 before your
echo   first render - the defaults in this repo assume a ~96 GB card.
echo.
pause
endlocal
exit /b 0

:piperr
echo.
echo !! Install hit an error above.
echo    If you are behind antivirus/corporate TLS inspection and see a
echo    CERTIFICATE_VERIFY_FAILED, see certs\README.md for the one-line fix.
pause
endlocal
exit /b 1
