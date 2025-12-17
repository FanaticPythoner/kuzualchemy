#!/usr/bin/env bash
set -Eeuo pipefail

# KuzuAlchemy Environment Setup
# Installs Python, Rust, and builds the rust extensions

# Usage:
#   ./setup_env.sh [VENV] [PY_REQ] [TOOLS_DIR] [PY_SPEC]
# Defaults:
VENV="${1:-.venv}"
PY_REQ="${2:-requirements.txt}"
TOOLS_DIR="${3:-.tools}"
PY_SPEC="${4:-3.12}"

log(){ printf '[kuzualchemy] %s\n' "$*"; }
die(){ printf '[kuzualchemy] ERROR: %s\n' "$*" >&2; exit 1; }
have(){ command -v "$1" >/dev/null 2>&1; }

# CRITICAL: Save original directory BEFORE any cd operations
ORIGINAL_DIR="$(pwd -P)"
ROOT="$ORIGINAL_DIR"
TOOLS="$ROOT/$TOOLS_DIR"
UV_HOME="$TOOLS/uv"
PY_HOME="$TOOLS/python"
mkdir -p "$UV_HOME" "$PY_HOME"

# ---------------- 1) Install uv LOCALLY (no PATH/profile edits) ----------------
export UV_NO_MODIFY_PATH=1
export UV_UNMANAGED_INSTALL="$UV_HOME"

if have curl; then
  log "Downloading uv (curl) -> $UV_HOME"
  curl -LsSf https://astral.sh/uv/install.sh | sh
elif have wget; then
  log "Downloading uv (wget) -> $UV_HOME"
  wget -qO- https://astral.sh/uv/install.sh | sh
else
  die "Need curl or wget to fetch uv."
fi

UV="$UV_HOME/uv"; [[ -x "$UV" ]] || UV="$UV_HOME/uv.exe"
[[ -x "$UV" ]] || die "uv not found in $UV_HOME"

# ---------------- 2) Install managed Python & create venv ----------------
export UV_PYTHON_INSTALL_DIR="$PY_HOME"
export UV_PYTHON_PREFERENCE="only-managed"
export UV_LINK_MODE="copy"  # Avoid hardlink warnings on WSL2/cross-filesystem

log "Installing managed Python $PY_SPEC under $PY_HOME"
"$UV" python install "$PY_SPEC" --force >/dev/null

log "Creating venv $VENV (seed pip)"
# Always overwrite without prompting
if [[ -d "$VENV" ]]; then
  log "Removing existing venv $VENV"
  rm -rf "$VENV"
fi
"$UV" venv "$VENV" --python "$PY_SPEC" --seed

# Compute venv executables (Git Bash on Windows is reported as MINGW/MSYS/CYGWIN)
UNAME="$(uname -s || true)"
if [[ "$UNAME" == MINGW* || "$UNAME" == MSYS* || "$UNAME" == CYGWIN* || "${OS:-}" == "Windows_NT" ]]; then
  VENV_PY="$ROOT/$VENV/Scripts/python.exe"
  IS_WINDOWS=1
else
  VENV_PY="$ROOT/$VENV/bin/python"
  IS_WINDOWS=0
fi

# Verify Python executable exists
if [[ ! -f "$VENV_PY" ]]; then
  die "venv python not found at $VENV_PY"
fi

# ---------------- 3) Install Python deps STRICTLY into the venv -------------
if [[ -f "$PY_REQ" ]]; then
  log "Installing Python requirements from $PY_REQ (venv pip)"
  "$VENV_PY" -m pip install --upgrade pip setuptools wheel
  "$VENV_PY" -m pip install -r "$PY_REQ"
else
  log "No $PY_REQ found; skipping Python deps"
fi

# ---------------- 3b) Install atp_pipeline wheel --------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ATP_WHL=""
ATP_WHL_TS=0
shopt -s nullglob
for f in "$SCRIPT_DIR"/atp_pipeline-*.whl; do
  [[ -f "$f" ]] || continue
  ts="$(stat -c '%Y' "$f" 2>/dev/null || stat -f '%m' "$f")"
  if (( ts > ATP_WHL_TS )); then
    ATP_WHL_TS="$ts"
    ATP_WHL="$f"
  fi
done
shopt -u nullglob
if [[ -n "$ATP_WHL" && -f "$ATP_WHL" ]]; then
  log "Installing atp_pipeline from local wheel: $(basename "$ATP_WHL")"
  "$VENV_PY" -m pip install --force-reinstall --no-deps "$ATP_WHL"
else
  log "⚠ atp_pipeline wheel not found under $SCRIPT_DIR (expected atp_pipeline-*.whl)"
fi

"$VENV_PY" -m pip install -e ".[dev,test]"

# ---------------- 4) Final Verification ----------------
log "Verifying installation..."

# Test Python
if "$VENV_PY" -c "import sys; sys.exit(0)" 2>/dev/null; then
  log "✓ Python is working ($VENV_PY)"
else
  log "⚠ Python verification failed for $VENV_PY"
  log "  Checking if file exists: $([ -f "$VENV_PY" ] && echo "YES" || echo "NO")"
  log "  Trying to run with full error:"
  "$VENV_PY" -c "import sys; print('Python OK'); sys.exit(0)" 2>&1 || true
fi

if "$VENV_PY" -c "import atp_pipeline, kuzualchemy; print('atp_pipeline:', atp_pipeline.__file__); print('kuzualchemy:', kuzualchemy.__file__)" >/dev/null; then
  log "✓ atp_pipeline and kuzualchemy are importable"
else
  die "Import check failed for atp_pipeline and/or kuzualchemy"
fi

echo
echo "=== KuzuAlchemy Environment Ready ==="
echo "venv:   $ROOT/$VENV"
echo "python: $VENV_PY"
echo
echo "To activate:"
echo "  source $ROOT/$VENV/bin/activate  # Linux/Mac"
echo "  $ROOT/$VENV/Scripts/activate     # Windows"
echo

log "Setup complete!"

# Final safety check - make sure we're back in the original directory
cd "$ORIGINAL_DIR"