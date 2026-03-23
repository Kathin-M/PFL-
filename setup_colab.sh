#!/usr/bin/env bash
# ============================================================
#  PFL – Google Colab Environment Setup
#  Run this once: !bash setup_colab.sh
# ============================================================
set -euo pipefail

echo "=============================="
echo " [1/5] Installing GHCup + GHC 9.2.8 + Cabal 3.6"
echo "=============================="
export BOOTSTRAP_HASKELL_NONINTERACTIVE=1
export BOOTSTRAP_HASKELL_GHC_VERSION=9.6.6
export BOOTSTRAP_HASKELL_CABAL_VERSION=3.10.3.0
export BOOTSTRAP_HASKELL_ADJUST_BASHRC=1

curl --proto '=https' --tlsv1.2 -sSf https://get-ghcup.haskell.org | sh

# Source ghcup env so ghc/cabal are on PATH for the rest of this script
export GHCUP_INSTALL_BASE_PREFIX="$HOME"
source "$HOME/.ghcup/env"

echo ""
echo "GHC version: $(ghc --version)"
echo "Cabal version: $(cabal --version | head -1)"

echo ""
echo "=============================="
echo " [2/5] Downloading Libtorch 2.0.1 (CPU, cxx11-abi, shared)"
echo "=============================="
LIBTORCH_URL="https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.0.1%2Bcpu.zip"
LIBTORCH_DIR="/usr/local/libtorch"

if [ ! -d "$LIBTORCH_DIR/lib" ]; then
  cd /tmp
  wget -q --show-progress -O libtorch.zip "$LIBTORCH_URL"
  unzip -q -o libtorch.zip -d /usr/local/
  rm libtorch.zip
  echo "Libtorch extracted to $LIBTORCH_DIR"
else
  echo "Libtorch already present at $LIBTORCH_DIR – skipping download"
fi

echo ""
echo "=============================="
echo " [3/5] Configuring environment variables"
echo "=============================="
export LD_LIBRARY_PATH="$LIBTORCH_DIR/lib:${LD_LIBRARY_PATH:-}"
export CPATH="$LIBTORCH_DIR/include:$LIBTORCH_DIR/include/torch/csrc/api/include:${CPATH:-}"
export LIBRARY_PATH="$LIBTORCH_DIR/lib:${LIBRARY_PATH:-}"

# Persist for future cells
cat >> "$HOME/.bashrc" <<'EOF'
export LD_LIBRARY_PATH="/usr/local/libtorch/lib:${LD_LIBRARY_PATH:-}"
export CPATH="/usr/local/libtorch/include:/usr/local/libtorch/include/torch/csrc/api/include:${CPATH:-}"
export LIBRARY_PATH="/usr/local/libtorch/lib:${LIBRARY_PATH:-}"
source "$HOME/.ghcup/env" 2>/dev/null || true
EOF

# Also write to a sourceable file for !cd ... && source env.sh && cabal ...
cat > /content/pfl_env.sh <<'EOF'
export LD_LIBRARY_PATH="/usr/local/libtorch/lib:${LD_LIBRARY_PATH:-}"
export CPATH="/usr/local/libtorch/include:/usr/local/libtorch/include/torch/csrc/api/include:${CPATH:-}"
export LIBRARY_PATH="/usr/local/libtorch/lib:${LIBRARY_PATH:-}"
source "$HOME/.ghcup/env" 2>/dev/null || true
EOF

# Update the dynamic linker cache
ldconfig

echo "LD_LIBRARY_PATH = $LD_LIBRARY_PATH"
echo "CPATH           = $CPATH"
echo "LIBRARY_PATH    = $LIBRARY_PATH"

echo ""
echo "=============================="
echo " [4/5] Updating Cabal package index"
echo "=============================="
cabal update

echo ""
echo "=============================="
echo " [5/5] Writing cabal.project.local (libtorch paths for libtorch-ffi)"
echo "=============================="
# This file goes into the cloned PFL- repo directory.
# We write it to /content so the user can copy it after cloning.
cat > /content/cabal.project.local <<EOF
package libtorch-ffi
  extra-include-dirs: /usr/local/libtorch/include
                    , /usr/local/libtorch/include/torch/csrc/api/include
  extra-lib-dirs:     /usr/local/libtorch/lib
EOF

echo ""
echo "============================================="
echo " ✅  Setup complete!"
echo "   GHC $(ghc --numeric-version) | Cabal $(cabal --numeric-version)"
echo "   Libtorch at $LIBTORCH_DIR"
echo ""
echo " Next steps:"
echo "   1. git clone https://github.com/Kathin-M/PFL-.git /content/PFL-"
echo "   2. cp /content/cabal.project.local /content/PFL-/"
echo "   3. cd /content/PFL- && source /content/pfl_env.sh && cabal build"
echo "   4. cabal run pfl-train"
echo "============================================="
