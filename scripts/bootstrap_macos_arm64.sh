#!/bin/bash
# Bootstrap script for macOS ARM64 (M1/M2/M3 Macs)
# Sets up a production-ready Python environment with proper ARM64 support
# Usage: bash scripts/bootstrap_macos_arm64.sh

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================================================
# Helper functions
# ============================================================================

info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

success() {
    echo -e "${GREEN}✅ $1${NC}"
}

warn() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

error() {
    echo -e "${RED}❌ $1${NC}"
    exit 1
}

# ============================================================================
# Preflight Checks
# ============================================================================

info "macOS ARM64 Bootstrap - Preflight Checks"
echo ""

# Check OS and architecture
MACHINE=$(uname -m)
SYSTEM=$(uname -s)

if [[ "$SYSTEM" != "Darwin" ]]; then
    error "This script is for macOS only. Detected: $SYSTEM"
fi

if [[ "$MACHINE" != "arm64" ]]; then
    warn "This script is optimized for ARM64 (M1/M2/M3). Detected: $MACHINE"
    warn "Install instructions for Intel Macs are in README.md"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        info "Exiting bootstrap."
        exit 0
    fi
fi

success "Running on ARM64 macOS ($MACHINE / $SYSTEM)"
echo ""

# ============================================================================
# Step 1: Check for Homebrew
# ============================================================================

info "Step 1/5: Checking Homebrew installation"

if ! command -v brew &> /dev/null; then
    warn "Homebrew not found. Installing..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

    # Add Homebrew to PATH
    eval "$(/opt/homebrew/bin/brew shellenv)"
fi

success "Homebrew is installed"
BREW_PATH=$(which brew)
info "Homebrew path: $BREW_PATH"
echo ""

# ============================================================================
# Step 2: Install or update pyenv
# ============================================================================

info "Step 2/5: Setting up pyenv (Python version manager)"

if ! command -v pyenv &> /dev/null; then
    warn "pyenv not found. Installing via Homebrew..."
    brew install pyenv

    # Add pyenv initialization to shell profile
    if [[ -f ~/.zprofile ]]; then
        if ! grep -q "pyenv init" ~/.zprofile; then
            echo 'eval "$(pyenv init --path)"' >> ~/.zprofile
            echo 'eval "$(pyenv init -)"' >> ~/.zprofile
            info "Added pyenv initialization to ~/.zprofile"
        fi
    fi

    if [[ -f ~/.bashrc ]]; then
        if ! grep -q "pyenv init" ~/.bashrc; then
            echo 'eval "$(pyenv init --path)"' >> ~/.bashrc
            echo 'eval "$(pyenv init -)"' >> ~/.bashrc
            info "Added pyenv initialization to ~/.bashrc"
        fi
    fi

    # Initialize pyenv for current session
    eval "$(pyenv init --path)"
    eval "$(pyenv init -)"
else
    success "pyenv is already installed"
fi

PYENV_PATH=$(which pyenv)
info "pyenv path: $PYENV_PATH"
echo ""

# ============================================================================
# Step 3: Install Python 3.11 with pyenv
# ============================================================================

info "Step 3/5: Installing Python 3.11 via pyenv"

PYTHON_VERSION="3.11"
INSTALLED_VERSION=$(pyenv versions | grep "^\* " | awk '{print $2}')

if [[ "$INSTALLED_VERSION" == "$PYTHON_VERSION"* ]]; then
    success "Python $INSTALLED_VERSION is already active"
else
    info "Installing Python $PYTHON_VERSION..."

    # Check available versions
    LATEST_311=$(pyenv install --list | grep "^[[:space:]]*$PYTHON_VERSION" | tail -1 | xargs)

    if [[ -z "$LATEST_311" ]]; then
        error "Could not find Python 3.11 in pyenv. Try: pyenv install --list | grep 3.11"
    fi

    info "Installing Python $LATEST_311..."
    pyenv install --skip-existing "$LATEST_311"

    # Set local version
    cd "$PROJECT_ROOT"
    pyenv local "$LATEST_311"

    success "Python $LATEST_311 installed and activated"
fi

PYTHON_PATH=$(which python3)
PYTHON_VERSION_STR=$(python3 --version)
success "$PYTHON_VERSION_STR at $PYTHON_PATH"
echo ""

# ============================================================================
# Step 4: Install uv (fast dependency manager)
# ============================================================================

info "Step 4/5: Installing uv (dependency manager)"

if ! command -v uv &> /dev/null; then
    warn "uv not found. Installing via Homebrew..."
    brew install uv
else
    success "uv is already installed"
fi

UV_PATH=$(which uv)
UV_VERSION=$(uv --version)
info "uv path: $UV_PATH (version: $UV_VERSION)"
echo ""

# ============================================================================
# Step 5: Install dependencies with uv
# ============================================================================

info "Step 5/5: Installing project dependencies with uv"

cd "$PROJECT_ROOT"

# Check if pyproject.toml exists
if [[ ! -f "pyproject.toml" ]]; then
    error "pyproject.toml not found in $PROJECT_ROOT"
fi

info "Installing base dependencies..."
uv sync

info "Installing development dependencies..."
uv sync --extra dev

success "All dependencies installed"
echo ""

# ============================================================================
# Verification
# ============================================================================

info "Verifying installation..."
echo ""

# Check Python path
echo "Python executable:"
python3 --version
which python3
echo ""

# Check architecture
echo "Architecture detection:"
python3 -c "import platform; print(f'  Machine: {platform.machine()}')"
python3 -c "import platform; print(f'  System: {platform.system()}')"
echo ""

# Try importing core modules
echo "Core module verification:"
python3 -c "import torch; print(f'  ✅ torch {torch.__version__}')" 2>/dev/null || echo "  ⚠️  torch not yet installed (will be installed on first use)"
python3 -c "import sentence_transformers; print(f'  ✅ sentence-transformers {sentence_transformers.__version__}')" 2>/dev/null || echo "  ⚠️  sentence-transformers not yet installed"
python3 -c "import numpy; print(f'  ✅ numpy {numpy.__version__}')" 2>/dev/null || echo "  ⚠️  numpy not installed"
python3 -c "import rank_bm25; print('  ✅ rank-bm25')" 2>/dev/null || echo "  ⚠️  rank-bm25 not installed"
python3 -c "import faiss; print('  ✅ faiss-cpu')" 2>/dev/null || echo "  ⚠️  faiss-cpu not installed (application has fallback)"
echo ""

# Check if MPS is available (only on macOS)
if command -v python3 &> /dev/null; then
    python3 << 'EOF'
import sys
try:
    import torch
    has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    if has_mps:
        print("🚀 Metal Performance Shaders (MPS) available for GPU acceleration")
    else:
        print("⚠️  Metal Performance Shaders (MPS) not available - will use CPU")
except ImportError:
    print("⚠️  torch not installed yet")
except Exception as e:
    print(f"⚠️  Could not check MPS: {e}")
EOF
fi
echo ""

# ============================================================================
# Next Steps
# ============================================================================

success "Bootstrap complete! 🎉"
echo ""
info "Next steps:"
echo "  1. (Optional) Activate the development environment:"
echo "     source .venv/bin/activate  # if uv created a virtual env"
echo ""
echo "  2. Verify with the doctor command:"
echo "     python3 -m clockify_rag.cli_modern doctor"
echo ""
echo "  3. Build the index:"
echo "     python3 -m clockify_rag.cli_modern ingest --input knowledge_helpcenter.md --force"
echo ""
echo "  4. Start the interactive chat:"
echo "     python3 -m clockify_rag.cli_modern chat"
echo ""
echo "📖 For more information, see:"
echo "   - README.md"
echo ""

info "Troubleshooting:"
echo "  - If torch/FAISS installation fails, see: README.md"
echo "  - If you see 'wrong architecture' errors, you may be running under Rosetta."
echo "    Check: python3 -c \"import platform; print(platform.machine())\""
echo "    Should show 'arm64', not 'x86_64'"
echo ""
