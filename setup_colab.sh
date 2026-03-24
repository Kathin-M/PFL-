#!/usr/bin/env bash
# ============================================================
#  PFL – Google Colab Environment Setup (FINAL)
#  Combines: cxx11-abi libtorch + shared linking + rpath
# ============================================================
set -euo pipefail

echo "=============================="
echo " [0/5] Installing system dependencies"
echo "=============================="
apt-get update -qq
apt-get install -y -qq build-essential curl libffi-dev libffi8ubuntu1 \
  libgmp-dev libgmp10 libncurses-dev libnuma-dev zlib1g-dev \
  pkg-config > /dev/null 2>&1
echo "System deps installed."

echo ""
echo "=============================="
echo " [1/5] Installing GHCup + GHC 9.6.6 + Cabal 3.10"
echo "=============================="
export BOOTSTRAP_HASKELL_NONINTERACTIVE=1
export BOOTSTRAP_HASKELL_GHC_VERSION=9.6.6
export BOOTSTRAP_HASKELL_CABAL_VERSION=3.10.3.0
export BOOTSTRAP_HASKELL_ADJUST_BASHRC=1

curl --proto '=https' --tlsv1.2 -sSf https://get-ghcup.haskell.org | sh
source "$HOME/.ghcup/env"

echo "GHC version: $(ghc --version)"
echo "Cabal version: $(cabal --version | head -1)"

echo ""
echo "=============================="
echo " [2/5] Downloading Libtorch 2.0.1 (CPU, cxx11-abi)"
echo "=============================="
LIBTORCH_URL="https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.0.1%2Bcpu.zip"
LIBTORCH_DIR="/usr/local/libtorch"

if [ ! -d "$LIBTORCH_DIR/lib" ]; then
  cd /tmp
  echo "Downloading libtorch (this takes ~1 min)..."
  wget -q --show-progress -O libtorch.zip "$LIBTORCH_URL"
  unzip -q -o libtorch.zip -d /usr/local/
  rm libtorch.zip
  echo "Libtorch extracted to $LIBTORCH_DIR"
else
  echo "Libtorch already present – skipping download"
fi

echo ""
echo "=============================="
echo " [3/5] Configuring linker and environment"
echo "=============================="
# Register libtorch with system dynamic linker
echo "/usr/local/libtorch/lib" > /etc/ld.so.conf.d/libtorch.conf
ldconfig 2>/dev/null || true

# Set environment variables
export LD_LIBRARY_PATH="$LIBTORCH_DIR/lib:${LD_LIBRARY_PATH:-}"
export CPATH="$LIBTORCH_DIR/include:$LIBTORCH_DIR/include/torch/csrc/api/include:${CPATH:-}"
export LIBRARY_PATH="$LIBTORCH_DIR/lib:${LIBRARY_PATH:-}"
export LIBTORCH_SKIP_DOWNLOAD=1

# Write env file for future cells
cat > /content/pfl_env.sh <<'EOF'
export PATH="$HOME/.ghcup/bin:$PATH"
source "$HOME/.ghcup/env" 2>/dev/null || true
export LD_LIBRARY_PATH="/usr/local/libtorch/lib:${LD_LIBRARY_PATH:-}"
export CPATH="/usr/local/libtorch/include:/usr/local/libtorch/include/torch/csrc/api/include:${CPATH:-}"
export LIBRARY_PATH="/usr/local/libtorch/lib:${LIBRARY_PATH:-}"
export LIBTORCH_SKIP_DOWNLOAD=1
EOF

echo "Environment configured."

echo ""
echo "=============================="
echo " [4/5] Writing cabal.project.local"
echo "=============================="
# Write to the PFL- repo directory if it exists, otherwise to /content
PROJ_DIR="/content/PFL-"
cat > "$PROJ_DIR/cabal.project.local" <<'EOF'
package libtorch-ffi
  extra-include-dirs: /usr/local/libtorch/include
                    , /usr/local/libtorch/include/torch/csrc/api/include
  extra-lib-dirs:     /usr/local/libtorch/lib
  ghc-options: -optl-Wl,-rpath,/usr/local/libtorch/lib

package hasktorch
  ghc-options: -optl-Wl,-rpath,/usr/local/libtorch/lib
EOF

echo "cabal.project.local written with rpath flags."

echo ""
echo "=============================="
echo " [5/5] Updating Cabal package index"
echo "=============================="
cabal update

echo ""
echo "============================================="
echo " ✅  Setup complete!"
echo "   GHC $(ghc --numeric-version) | Cabal $(cabal --numeric-version)"
echo "   Libtorch at $LIBTORCH_DIR (cxx11-abi)"
echo ""
echo " Next:"
echo "   source /content/pfl_env.sh"
echo "   cd /content/PFL- && cabal build"
echo "============================================="
