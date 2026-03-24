#!/usr/bin/env bash
# ============================================================
#  PFL – Google Colab Environment Setup
#  Run this once: !bash setup_colab.sh
# ============================================================
set -euo pipefail

echo "=============================="
echo " [0/4] Installing system dependencies"
echo "=============================="
apt-get update -qq
apt-get install -y -qq libgmp-dev libnuma-dev zlib1g-dev libffi-dev > /dev/null 2>&1
echo "System deps installed."

echo ""
echo "=============================="
echo " [1/4] Installing GHCup + GHC 9.6.6 + Cabal 3.10"
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
echo " [2/4] Configuring libtorch-ffi environment"
echo "=============================="
# libtorch-ffi 2.0.1.10 will automatically download libtorch
# into its cache directory. We just configure the version & flavor.
export LIBTORCH_VERSION=2.0.1
export LIBTORCH_CUDA_VERSION=cpu

echo "LIBTORCH_VERSION      = $LIBTORCH_VERSION"
echo "LIBTORCH_CUDA_VERSION = $LIBTORCH_CUDA_VERSION"
echo "libtorch-ffi will auto-download libtorch during cabal build."

# Persist env vars for future cells
cat > /content/pfl_env.sh <<'EOF'
export PATH="$HOME/.ghcup/bin:$PATH"
source "$HOME/.ghcup/env" 2>/dev/null || true
export LIBTORCH_VERSION=2.0.1
export LIBTORCH_CUDA_VERSION=cpu
EOF

echo ""
echo "=============================="
echo " [3/4] Updating Cabal package index"
echo "=============================="
cabal update

echo ""
echo "=============================="
echo " [4/4] Setup complete!"
echo "=============================="
echo "   GHC $(ghc --numeric-version) | Cabal $(cabal --numeric-version)"
echo ""
echo " libtorch-ffi will automatically download libtorch $LIBTORCH_VERSION (CPU)"
echo " during the first 'cabal build'."
echo ""
echo " Next steps:"
echo "   1. source /content/pfl_env.sh"
echo "   2. cd /content/PFL- && cabal build"
echo "   3. cabal run pfl-train"
echo "============================================="
