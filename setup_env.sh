#!/usr/bin/env bash
set -Eeuo pipefail

# KuzuAlchemy Environment Setup
# Installs Python, Rust, and builds the rust extensions

# Usage:
#   ./setup_env.sh [VENV] [TOOLS_DIR] [PY_SPEC]
# Defaults:
VENV="${1:-.venv}"
TOOLS_DIR="${2:-.tools}"
PY_SPEC="${3:-cpython-3.14.5+freethreaded}"
PY_VERSION="${PY_VERSION:-3.14.5}"
PY_SOURCE_SHA256="${PY_SOURCE_SHA256:-7e32597b99e5d9a39abed35de4693fa169df3e5850d4c334337ffd6a19a36db6}"
PY_REQUIRED_MODULES="${PY_REQUIRED_MODULES:-_bz2 _ctypes _curses _hashlib _lzma _sqlite3 _ssl _uuid _zstd readline zlib}"
PY_STANDALONE_RELEASE="${PY_STANDALONE_RELEASE:-20260510}"
PY_STANDALONE_SHA256_X86_64="${PY_STANDALONE_SHA256_X86_64:-659827e25d43d3579f074393dfefed33625c338bad211a78f25850165e9ea41f}"

log(){ printf '[kuzualchemy] %s\n' "$*"; }
die(){ printf '[kuzualchemy] ERROR: %s\n' "$*" >&2; exit 1; }
have(){ command -v "$1" >/dev/null 2>&1; }

ensure_free_threaded_python(){
  python_executable="$1"
  [[ -x "$python_executable" ]] || return 1
  if ! PY_REQUIRED_MODULES="$PY_REQUIRED_MODULES" PYTHON_GIL=0 "$python_executable" - <<'PY'
import sys
import sysconfig

if sys.implementation.name != "cpython":
    raise SystemExit("expected CPython")
if sys.version_info < (3, 14, 5):
    raise SystemExit("expected CPython >=3.14.5")
if sysconfig.get_config_var("Py_GIL_DISABLED") != 1:
    raise SystemExit("expected Py_GIL_DISABLED=1")
gil_probe = getattr(sys, "_is_gil_enabled", None)
if gil_probe is None or gil_probe():
    raise SystemExit("expected disabled GIL")
import os
for module_name in os.environ["PY_REQUIRED_MODULES"].split():
    __import__(module_name)
PY
  then
    return 1
  fi
}

python_platform_tag(){
  printf 'linux-%s-gnu\n' "$(uname -m)"
}

python_source_prefix(){
  printf '%s/cpython-%s+freethreaded-%s\n' "$PY_HOME" "$PY_VERSION" "$(python_platform_tag)"
}

install_python_from_standalone(){
  machine="$(uname -m)"
  case "$machine" in
    x86_64) checksum="$PY_STANDALONE_SHA256_X86_64" ;;
    *) return 1 ;;
  esac
  asset_name="cpython-$PY_VERSION+$PY_STANDALONE_RELEASE-${machine}-unknown-linux-gnu-freethreaded-install_only_stripped.tar.gz"
  temp_dir="$PY_HOME/.temp"
  archive="$temp_dir/$asset_name"
  extracted="$temp_dir/${asset_name%.tar.gz}"
  prefix="$(python_source_prefix)"
  python_bin="$prefix/bin/python3.14t"
  mkdir -p "$temp_dir"
  if [[ ! -f "$archive" ]]; then
    curl -fL "https://github.com/astral-sh/python-build-standalone/releases/download/$PY_STANDALONE_RELEASE/$asset_name" -o "$archive" || return 1
  fi
  printf '%s  %s\n' "$checksum" "$archive" | sha256sum -c - || return 1
  rm -rf "$extracted" "$prefix"
  mkdir -p "$extracted"
  tar -C "$extracted" -xzf "$archive" || return 1
  [[ -d "$extracted/python" ]] || return 1
  mv "$extracted/python" "$prefix"
  ln -sf python3.14t "$prefix/bin/python3.14"
  ln -sf python3.14 "$prefix/bin/python3"
  ln -sf python3.14 "$prefix/bin/python"
  ensure_free_threaded_python "$python_bin" || return 1
  printf '%s\n' "$python_bin"
}

install_python_from_source(){
  prefix="$(python_source_prefix)"
  python_bin="$prefix/bin/python3.14t"
  if ensure_free_threaded_python "$python_bin"; then
    printf '%s\n' "$python_bin"
    return 0
  fi
  for tool in curl tar xz make gcc sha256sum; do
    have "$tool" || return 1
  done
  temp_dir="$PY_HOME/.temp"
  tarball="$temp_dir/Python-$PY_VERSION.tar.xz"
  source_dir="$temp_dir/Python-$PY_VERSION"
  mkdir -p "$temp_dir"
  if [[ ! -f "$tarball" ]]; then
    curl -fL "https://www.python.org/ftp/python/$PY_VERSION/Python-$PY_VERSION.tar.xz" -o "$tarball" || return 1
  fi
  printf '%s  %s\n' "$PY_SOURCE_SHA256" "$tarball" | sha256sum -c - || return 1
  rm -rf "$source_dir" "$prefix"
  tar -C "$temp_dir" -xf "$tarball" || return 1
  jobs="$(getconf _NPROCESSORS_ONLN 2>/dev/null || printf '1')"
  (
    set -e
    cd "$source_dir"
    LDFLAGS="${LDFLAGS:-} -Wl,-rpath,$prefix/lib" \
      ./configure \
      --prefix="$prefix" \
      --disable-gil \
      --enable-shared \
      --with-ensurepip=install
    make -j"$jobs"
    make install
  ) || return 1
  ln -sf python3.14t "$prefix/bin/python3.14"
  ln -sf python3.14 "$prefix/bin/python3"
  ln -sf python3.14 "$prefix/bin/python"
  if [[ -x "$prefix/bin/python3.14t-config" ]]; then
    ln -sf python3.14t-config "$prefix/bin/python3.14-config"
    ln -sf python3.14-config "$prefix/bin/python3-config"
  fi
  ensure_free_threaded_python "$python_bin" || return 1
  printf '%s\n' "$python_bin"
}

# Capture root before directory changes.
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
export PYTHON_GIL=0

log "Installing managed Python $PY_SPEC under $PY_HOME"
PYTHON_FOR_VENV="$PY_SPEC"
if ! "$UV" python install "$PY_SPEC" --force >/dev/null; then
  PYTHON_FOR_VENV="$(install_python_from_standalone || install_python_from_source)"
fi

log "Creating venv $VENV (seed pip)"
# Always overwrite without prompting
if [[ -d "$VENV" ]]; then
  log "Removing existing venv $VENV"
  rm -rf "$VENV"
fi
"$UV" venv "$VENV" --python "$PYTHON_FOR_VENV" --seed

# Compute venv executables (Git Bash on Windows is reported as MINGW/MSYS/CYGWIN)
UNAME="$(uname -s)"
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
ensure_free_threaded_python "$VENV_PY" || die "venv Python is not CPython >=3.14.5 free-threaded with disabled GIL: $VENV_PY"

# ---------------- 3) Install Python deps STRICTLY into the venv -------------
# Dependencies and wheels are now handled by pyproject.toml
log "Installing project and dependencies from pyproject.toml"
"$VENV_PY" -m pip install --upgrade pip setuptools wheel
"$VENV_PY" -m pip install -e ".[dev,test]"

# ---------------- 4) Final Verification ----------------
log "Verifying installation..."

# Test Python
if "$VENV_PY" -c "import sys; sys.exit(0)" 2>/dev/null; then
  log "Python is working ($VENV_PY)"
else
  die "Python verification failed for $VENV_PY"
fi

if "$VENV_PY" -c "import atp_pipeline, kuzualchemy; print('atp_pipeline:', atp_pipeline.__file__); print('kuzualchemy:', kuzualchemy.__file__)" >/dev/null; then
  log "atp_pipeline and kuzualchemy are importable"
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

log "Setup complete"

# Return to the original directory.
cd "$ORIGINAL_DIR"
