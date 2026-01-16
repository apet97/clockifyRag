#!/bin/bash
# Clockify RAG CLI - Automated Setup Script
# One-command setup: ./setup.sh
#
# This script:
# 1. Checks system requirements (Python 3.11-3.13, curl)
# 2. Creates Python virtual environment
# 3. Installs dependencies
# 4. Checks Ollama connectivity
# 5. Optionally installs pre-commit hooks

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Banner
echo "======================================================================"
echo "  Clockify RAG CLI - Automated Setup"
echo "======================================================================"
echo ""

# Step 1: Check Python version
info "Checking Python version..."
if ! command -v python3 &> /dev/null; then
    error "python3 not found. Please install Python 3.11-3.13."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 11 ]); then
    error "Python 3.11+ required. Found: $PYTHON_VERSION"
    exit 1
fi

if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 14 ]; then
    error "Python 3.11-3.13 required. Found: $PYTHON_VERSION"
    exit 1
fi

success "Python $PYTHON_VERSION detected"

# Step 2: Check curl
info "Checking curl availability..."
if ! command -v curl &> /dev/null; then
    warning "curl not found. Ollama check will be skipped."
    SKIP_OLLAMA_CHECK=1
else
    success "curl found"
fi

# Step 3: Create virtual environment
info "Creating virtual environment (rag_env)..."
if [ -d "rag_env" ]; then
    warning "Virtual environment already exists. Skipping creation."
else
    python3 -m venv rag_env
    success "Virtual environment created"
fi

# Step 4: Activate virtual environment
info "Activating virtual environment..."
source rag_env/bin/activate || {
    error "Failed to activate virtual environment"
    exit 1
}
success "Virtual environment activated"

# Step 5: Upgrade pip
info "Upgrading pip..."
python -m pip install --upgrade pip -q
success "pip upgraded"

# Step 5.5: Check for M1 and handle conda vs pip
MACHINE_ARCH=$(uname -m 2>/dev/null || echo "unknown")
SYSTEM_OS=$(uname -s 2>/dev/null || echo "unknown")
IS_M1_MAC=false

if [ "$SYSTEM_OS" = "Darwin" ] && [ "$MACHINE_ARCH" = "arm64" ]; then
    IS_M1_MAC=true
    echo ""
    warning "Apple Silicon (M1/M2/M3) detected!"
    echo ""
    echo "  FAISS on ARM64 requires special handling."
    echo "  Choose installation method:"
    echo ""
    echo "  [1] Conda (RECOMMENDED)"
    echo "      • FAISS ARM64 builds from conda-forge"
    echo "      • PyTorch with MPS GPU acceleration"
    echo "      • Best compatibility and performance"
    echo ""
    echo "  [2] pip (NOT RECOMMENDED)"
    echo "      • FAISS may fail to install"
    echo "      • No GPU acceleration"
    echo "      • Fallback to USE_ANN=none if build fails"
    echo ""
    read -p "Choose installation method [1=conda, 2=pip, q=quit]: " -n 1 -r
    echo

    if [[ $REPLY == "1" ]]; then
        # Conda flow for M1
        echo ""
        info "Using Conda installation for M1..."
        info "Please ensure Miniforge/Conda is installed:"
        echo "    brew install miniforge"
        echo ""

        # Deactivate current venv if active
        if [ -n "$VIRTUAL_ENV" ]; then
            info "Deactivating current virtual environment..."
            deactivate || true
        fi

        # Check if conda is available
        if ! command -v conda &> /dev/null; then
            error "conda not found. Install Miniforge first:"
            echo "    /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
            echo "    brew install miniforge"
            echo "    conda init"
            exit 1
        fi

        # Create conda environment
        CONDA_ENV_NAME="clockify_rag"
        info "Creating conda environment: $CONDA_ENV_NAME..."

        # Check if environment already exists
        if conda env list | grep -q "^$CONDA_ENV_NAME "; then
            warning "Conda environment '$CONDA_ENV_NAME' already exists"
            read -p "Recreate it? [y/N]: " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                conda env remove -n "$CONDA_ENV_NAME" -y
                conda create -n "$CONDA_ENV_NAME" python=3.11 -y
            fi
        else
            conda create -n "$CONDA_ENV_NAME" python=3.11 -y
        fi
        success "Conda environment created/verified"

        # Activate conda environment
        info "Activating conda environment..."
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate "$CONDA_ENV_NAME"
        success "Conda environment activated"

        # Install FAISS via conda-forge
        info "Installing FAISS from conda-forge (native ARM64 build)..."
        conda install -c conda-forge faiss-cpu=1.8.0 -y
        success "FAISS installed"

        # Install NumPy via conda
        info "Installing NumPy via conda..."
        conda install -c conda-forge numpy -y
        success "NumPy installed"

        # Install PyTorch with MPS support
        info "Installing PyTorch with MPS GPU support..."
        conda install -c pytorch pytorch -y
        success "PyTorch installed"

        # Install SentenceTransformers
        info "Installing SentenceTransformers..."
        conda install -c conda-forge sentence-transformers -y
        success "SentenceTransformers installed"

        # Upgrade pip
        info "Upgrading pip..."
        python -m pip install --upgrade pip -q
        success "pip upgraded"

        # Install via pip (FAISS already excluded from pyproject.toml, no --no-deps needed)
        info "Installing remaining dependencies via pip..."
        pip install -e .
        success "Remaining dependencies installed"

        echo ""
        success "M1 Conda setup complete!"
        echo ""
        echo "To activate this environment in future sessions, run:"
        echo "    conda activate $CONDA_ENV_NAME"
        echo ""

        # Exit after conda setup
        exit 0

    elif [[ $REPLY == "2" ]]; then
        # pip flow (not recommended)
        warning "Using pip installation (not recommended for M1)..."
        warning "FAISS may fail. If it does, re-run setup and choose conda instead."
        echo ""

    elif [[ $REPLY == "q" ]]; then
        info "Setup cancelled."
        exit 0
    else
        error "Invalid choice. Please run setup.sh again."
        exit 1
    fi
    echo ""
fi

# Step 6: Install dependencies (pip path)
info "Installing dependencies (this may take a few minutes)..."
if [ -f "requirements.lock" ]; then
    info "Installing from requirements.lock (pinned versions)..."
    pip install -r requirements.lock -q
else
    info "Installing from requirements.txt..."
    pip install -r requirements.txt -q
fi
success "Dependencies installed"

# Step 7: Check Ollama connectivity
if [ -z "$SKIP_OLLAMA_CHECK" ]; then
    info "Checking Ollama connectivity..."
    OLLAMA_URL="${OLLAMA_URL:-http://127.0.0.1:11434}"
    if curl -sf "$OLLAMA_URL/api/version" > /dev/null 2>&1; then
        OLLAMA_VERSION=$(curl -s "$OLLAMA_URL/api/version" | grep -o '"version":"[^"]*"' | cut -d'"' -f4 || echo "unknown")
        success "Ollama is running (version: $OLLAMA_VERSION)"

        # Check for required models
        info "Checking for required Ollama models..."

        # Check embedding model
        if ollama list 2>/dev/null | grep -q "nomic-embed-text"; then
            success "Embedding model (nomic-embed-text) found"
        else
            warning "Embedding model (nomic-embed-text) not found"
            echo "  To install: ollama pull nomic-embed-text"
        fi

        # Check generation model
        if ollama list 2>/dev/null | grep -q "qwen2.5:32b"; then
            success "Generation model (qwen2.5:32b) found"
        else
            warning "Generation model (qwen2.5:32b) not found"
            echo "  To install: ollama pull qwen2.5:32b"
            echo "  Or use smaller model: ollama pull qwen2.5:7b"
        fi
    else
        warning "Ollama is not running or not reachable at $OLLAMA_URL"
        echo "  Please start Ollama: ollama serve"
        echo "  Or install from: https://ollama.com"
    fi
fi

# Step 8: Optional pre-commit hooks
echo ""
read -p "Install pre-commit git hooks? (recommended for contributors) [y/N]: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    info "Installing pre-commit hooks..."
    pre-commit install
    success "Pre-commit hooks installed"
else
    info "Skipping pre-commit hooks"
fi

# Step 9: Summary
echo ""
echo "======================================================================"
success "Setup complete!"
echo "======================================================================"
echo ""
echo "Next steps:"
echo "  1. Activate virtual environment:"
echo "     $ source rag_env/bin/activate"
echo ""
echo "  2. Build knowledge base (first time only):"
echo "     $ make build"
echo ""
echo "  3. Start interactive chat:"
echo "     $ make chat"
echo ""
echo "  4. Run tests:"
echo "     $ make test"
echo ""
echo "For more commands, run:"
echo "  $ make help"
echo ""
echo "Documentation:"
echo "  - README.md"
echo ""
