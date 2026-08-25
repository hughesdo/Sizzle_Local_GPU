#!/usr/bin/env bash
# ===========================================================================
#  SIZZLE - first-time setup for Linux / macOS.  ./install.sh
# ===========================================================================
#  The POSIX counterpart of install.bat. Same eight steps, same results.
#  This is the one you want on a rented/hosted GPU box, which is almost always
#  Linux.
#
#  Builds a local Python 3.12 environment with CUDA PyTorch, installs
#  Lightricks' LTX-2 inference packages, and creates your .env.
#  Safe to re-run: it reuses an existing .venv and won't overwrite your .env.
#
#  It does NOT download the ~128 GB of model weights - that is a separate,
#  resumable step it offers you at the end (see models/README.md).
#
#  ---------------------------------------------------------------------------
#  GPU PORTABILITY - the one thing you may need to change
#  ---------------------------------------------------------------------------
#  Step 6 installs torch from PyTorch's CUDA index, NOT from PyPI (PyPI's
#  default torch is CPU-only and would silently give you a box that cannot
#  render). It defaults to cu132 because Blackwell (sm_120) requires it; cu132
#  also covers every older CUDA card, so this is usually right.
#
#  If nvidia-smi reports a CUDA version below 13, override without editing:
#      SIZZLE_TORCH_INDEX=https://download.pytorch.org/whl/cu128 \
#      SIZZLE_TORCH_SPEC=torch ./install.sh
#
#  See SETUP.md section 3.1 for the driver -> wheel table.
# ===========================================================================
set -uo pipefail
cd "$(dirname "$0")"

# --- GPU-portability knobs (override from the environment) -----------------
TORCH_INDEX="${SIZZLE_TORCH_INDEX:-https://download.pytorch.org/whl/cu132}"
TORCH_SPEC="${SIZZLE_TORCH_SPEC:-torch==2.13.0+cu132}"
# torchaudio for cu132 currently only exists on the test index. If you moved
# TORCH_INDEX to a stable cuXXX channel, point this at the same channel.
TORCHAUDIO_INDEX="${SIZZLE_TORCHAUDIO_INDEX:-https://download.pytorch.org/whl/test/cu132/}"
PYTHON_VERSION="${SIZZLE_PYTHON_VERSION:-3.12}"

say()  { printf '\n>> %s\n' "$*"; }
warn() { printf '\n!! %s\n' "$*" >&2; }
die()  { warn "$*"; exit 1; }

echo
echo " ========================================================="
echo "  SIZZLE local-GPU setup"
echo " ========================================================="

# --- 1. uv (installs and manages Python for us) ----------------------------
say "Checking for uv..."
if ! command -v uv >/dev/null 2>&1; then
  echo "   uv not found - installing it..."
  curl -LsSf https://astral.sh/uv/install.sh | sh || die "Could not install uv. See https://docs.astral.sh/uv/getting-started/installation/"
  # the installer drops uv here but does not touch the CURRENT shell's PATH
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
  command -v uv >/dev/null 2>&1 || die "uv installed but not on PATH. Open a NEW shell and re-run ./install.sh"
fi

# --- 2. ffmpeg -------------------------------------------------------------
# Needed for the final mux, and for SIZZLE_BACKEND=mock. Minimal CUDA container
# images almost never ship it.
say "Checking for ffmpeg..."
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "   ffmpeg not found - trying your package manager..."
  if   command -v apt-get >/dev/null 2>&1; then sudo apt-get update -qq && sudo apt-get install -y ffmpeg
  elif command -v dnf     >/dev/null 2>&1; then sudo dnf install -y ffmpeg
  elif command -v yum     >/dev/null 2>&1; then sudo yum install -y ffmpeg
  elif command -v pacman  >/dev/null 2>&1; then sudo pacman -S --noconfirm ffmpeg
  elif command -v brew    >/dev/null 2>&1; then brew install ffmpeg
  else
    warn "No known package manager. Install ffmpeg yourself, ideally an NVENC build:
       curl -L https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz | tar xJ
     then put it on PATH. Or set SIZZLE_NVENC=0 and use any build."
  fi
fi

# --- 3. NVIDIA driver sanity check -----------------------------------------
say "Checking for an NVIDIA GPU..."
if ! command -v nvidia-smi >/dev/null 2>&1; then
  warn "nvidia-smi not found. Sizzle renders on an NVIDIA GPU via CUDA.
     Install a current NVIDIA driver, or run with SIZZLE_BACKEND=mock to
     explore the app without a model. Continuing anyway..."
else
  nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv
  echo "   (SETUP.md section 4 maps VRAM -> the settings you need)"
fi

# --- 4. the LTX-2 inference packages ---------------------------------------
#  Sizzle drives Lightricks' own pipeline code. We vendor it as a sibling
#  checkout and pip-install it editable, so upgrading is just a git pull.
#  The exact validated commit is recorded in VENDOR-PINS.txt - check it out if
#  upstream HEAD ever breaks the import in backend/ltx_engine.py.
if [ ! -f "LTX-2/packages/ltx-pipelines/pyproject.toml" ]; then
  say "Cloning Lightricks/LTX-2 (inference pipelines)..."
  command -v git >/dev/null 2>&1 || die "git not found. Install git, then re-run ./install.sh"
  git clone --depth 1 https://github.com/Lightricks/LTX-2.git LTX-2 || die "Failed to clone LTX-2."
else
  say "LTX-2 checkout already present"
fi

# --- 5. Python 3.12 + venv --------------------------------------------------
say "Installing Python ${PYTHON_VERSION} (via uv)..."
uv python install "${PYTHON_VERSION}"

if [ ! -x ".venv/bin/python" ]; then
  say "Creating virtual environment (.venv)..."
  uv venv --python "${PYTHON_VERSION}" .venv || die "Could not create .venv"
else
  say "Reusing existing .venv"
fi
PY=".venv/bin/python"

# --- 6. CUDA PyTorch --------------------------------------------------------
#  ~3 GB download the first time. See the header for how to retarget this.
say "Installing PyTorch from ${TORCH_INDEX} (large download, be patient)..."
uv pip install --python "$PY" --index-url "$TORCH_INDEX" "$TORCH_SPEC" || die "torch install failed - see the TLS/CERTIFICATE note in certs/README.md, or retarget SIZZLE_TORCH_INDEX (SETUP.md 3.1)."
uv pip install --python "$PY" --index-url "$TORCHAUDIO_INDEX" torchaudio || die "torchaudio install failed."

# --- 7. LTX packages + Sizzle's own deps -----------------------------------
#  Only ltx-core and ltx-pipelines. NOT ltx-kernels (optional fused CUDA
#  kernels, builds from source) and NOT ltx-trainer (training only).
say "Installing LTX-2 inference packages..."
uv pip install --python "$PY" -e LTX-2/packages/ltx-core -e LTX-2/packages/ltx-pipelines || die "LTX package install failed."

say "Installing Sizzle dependencies..."
uv pip install --python "$PY" -r requirements.txt || die "Dependency install failed."

# --- 8. .env ----------------------------------------------------------------
#  In the backup repo .env is committed, so this is normally a no-op - which is
#  intended. Secrets belong in .env.local (gitignored), NOT here. See SETUP.md 7.
if [ ! -f ".env" ]; then
  say "Creating .env from .env.example"
  cp .env.example .env
else
  say ".env already exists - leaving it as-is"
fi

# --- 9. verify --------------------------------------------------------------
say "Verifying the install..."
"$PY" -c "import torch; print('   torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
"$PY" -c "from ltx_pipelines.a2vid_two_stage import A2VidPipelineTwoStage; print('   LTX-2 audio-to-video pipeline: OK')" || die "LTX-2 pipeline import failed. Upstream API may have drifted - try the pinned commit in VENDOR-PINS.txt."

echo
echo " ========================================================="
echo "  Setup complete."
echo " ========================================================="
echo
echo "  Still needed: the model weights (~128 GB, resumable)."
echo "  On a HOSTED box, put them on the persistent volume first:"
echo "      export LTX_MODEL_ROOT=/workspace/ltx-weights"
echo
read -r -p "  Download them now? [y/N] " DLNOW
case "${DLNOW:-n}" in
  [yY]*) "$PY" scripts/download_models.py ;;
  *)     echo; echo "  Run this when you're ready:"; echo "      $PY scripts/download_models.py" ;;
esac

cat <<'EOF'

  Then:
    ./run.sh      run it locally at http://127.0.0.1:8000
    ./host.sh     share it over a Cloudflare tunnel

  NOTE: the Gemma text encoder is a GATED download. Accept the license at
    https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized
  then run:  .venv/bin/hf auth login
  Full details: models/README.md

  Rebuilding on different hardware? Read SETUP.md section 4 before your first
  render - the defaults in this repo assume a ~96 GB card.

EOF
