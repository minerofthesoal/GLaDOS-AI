#!/usr/bin/env bash
# ============================================================
#  GLaDOS Assistant – Setup Script
#  Supports: Arch Linux  |  Linux Mint
#  Uses system Python 3.12 (with PyTorch already installed)
# ============================================================
set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info()  { echo -e "${GREEN}[setup]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[warn]${NC}   $*"; }
err()   { echo -e "${RED}[error]${NC}  $*" >&2; }
step()  { echo -e "\n${CYAN}══ $* ══${NC}"; }

# ── Find python3.12 ─────────────────────────────────────────
step "Locating Python 3.12"
if ! command -v python3.12 &>/dev/null; then
    err "python3.12 not found on PATH."
    exit 1
fi
info "Found: $(python3.12 --version) at $(command -v python3.12)"

# Verify torch is accessible from system python3.12
if ! python3.12 -c "import torch" 2>/dev/null; then
    err "torch is not importable from python3.12."
    err "Make sure PyTorch is installed for your system Python 3.12."
    exit 1
fi
TORCH_VER=$(python3.12 -c "import torch; print(torch.__version__)")
info "torch $TORCH_VER found in system Python 3.12."

# ── System packages ─────────────────────────────────────────
step "Installing system packages"
if command -v pacman &>/dev/null; then
    sudo pacman -S --needed --noconfirm \
        alsa-utils espeak-ng wget ffmpeg
else
    sudo apt-get update -qq
    sudo apt-get install -y --no-install-recommends \
        alsa-utils espeak-ng wget ffmpeg \
        python3-dev build-essential
fi

# ── venv using system python3.12 with access to system packages ──
step "Creating venv (system-packages passthrough)"
VENV_DIR="$(pwd)/glados_venv"
if [ -d "$VENV_DIR" ]; then
    warn "Removing old venv to recreate with correct Python…"
    rm -rf "$VENV_DIR"
fi

# --system-site-packages lets the venv see torch from system Python 3.12
python3.12 -m venv --system-site-packages "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
info "Venv active: $VIRTUAL_ENV"
info "Python: $(python --version)"

python -m pip install --upgrade pip --quiet

# ── HuggingFace stack ────────────────────────────────────────
step "HuggingFace / Transformers"
pip install \
    "transformers>=4.44.0" \
    "accelerate>=0.30.0" \
    "huggingface-hub>=0.24.0" \
    sentencepiece \
    protobuf \
    --quiet
info "Transformers stack installed."

# ── Piper TTS ────────────────────────────────────────────────
step "Piper TTS"
pip install piper-tts --quiet
command -v piper &>/dev/null \
    && info "piper at: $(command -v piper)" \
    || warn "piper not on PATH – glados_assistant.py will locate it in the venv."

# ── sounddevice ─────────────────────────────────────────────
step "Audio I/O"
pip install sounddevice numpy --quiet \
    && info "sounddevice installed." \
    || warn "sounddevice failed – mic input will fall back to arecord."

# ── Verify ───────────────────────────────────────────────────
step "Verification"
python - <<'PYCHECK'
import torch, sys
print(f"  Python  : {sys.version.split()[0]}")
print(f"  PyTorch : {torch.__version__}")
if torch.cuda.is_available():
    d = torch.cuda.get_device_properties(0)
    print(f"  GPU     : {d.name}  ({d.total_memory // 1024**2} MB)  sm_{d.major}{d.minor}")
    print(f"  Status  : OK ✓")
else:
    print("  CUDA    : not available (CPU mode)")
PYCHECK

# ── Done ─────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}╔══════════════════════════════════╗${NC}"
echo -e "${GREEN}║  Setup complete!  ✓              ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════╝${NC}"
echo ""
echo -e "  ${YELLOW}source $(pwd)/glados_venv/bin/activate${NC}"
echo -e "  ${YELLOW}python glados_assistant.py${NC}"
echo -e "  ${YELLOW}python glados_assistant.py --voice${NC}"
echo ""

# ── 3D viewer deps (replaces Panda3D) ───────────────────────
step "Installing 3D viewer dependencies (trimesh + pyglet + PyOpenGL)"
pip install trimesh pyglet PyOpenGL PyOpenGL_accelerate --quiet \
    && info "3D deps installed." \
    || warn "3D deps failed — viewer won't work but assistant will."
