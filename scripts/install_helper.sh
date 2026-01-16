#!/bin/bash
# Installation helper for macOS M1/M2/M3
# Detects platform and recommends best installation method
#
# Usage: ./scripts/install_helper.sh

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "====================================="
echo "  Clockify RAG CLI - Install Helper"
echo "====================================="
echo ""

# Detect platform
PLATFORM=$(uname -m 2>/dev/null || echo "unknown")
SYSTEM=$(uname -s 2>/dev/null || echo "unknown")

echo "Detected platform: $SYSTEM $PLATFORM"
echo ""

if [ "$SYSTEM" = "Darwin" ] && [ "$PLATFORM" = "arm64" ]; then
    echo -e "${GREEN}✅ Apple Silicon (M1/M2/M3) detected${NC}"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  RECOMMENDED INSTALLATION METHOD: Conda"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo -e "${BLUE}Why conda?${NC}"
    echo "  • FAISS ARM64 builds available via conda-forge"
    echo "  • PyTorch with MPS acceleration (Apple GPU)"
    echo "  • Better package compatibility on Apple Silicon"
    echo "  • Avoids common pip installation failures"
    echo ""
    echo -e "${BLUE}Quick start (conda):${NC}"
    echo ""
    echo "  # 1. Install Miniforge (conda for M1)"
    echo "  brew install miniforge"
    echo ""
    echo "  # 2. Create environment"
    echo "  conda create -n rag_env python=3.11"
    echo "  conda activate rag_env"
    echo ""
    echo "  # 3. Install dependencies (one-line command)"
    echo "  conda install -c conda-forge faiss-cpu=1.8.0 numpy requests && \\"
    echo "  conda install -c pytorch pytorch sentence-transformers && \\"
    echo "  pip install urllib3==2.2.3 rank-bm25==0.2.2 tiktoken nltk"
    echo ""
    echo "  # 4. Verify installation"
    echo "  python3 -c \"import numpy, faiss, torch, sentence_transformers; print('✅ All OK')\""
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo -e "${YELLOW}Alternative: pip (not recommended for M1)${NC}"
    echo "  ./setup.sh"
    echo ""
    echo "  ${YELLOW}Warning:${NC} FAISS may fail to install via pip on M1."
    echo "  If pip fails, use conda method above."
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    read -p "Would you like to view detailed M1 instructions? (requirements-m1.txt) [y/N]: " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        if [ -f "requirements-m1.txt" ]; then
            echo ""
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            cat requirements-m1.txt
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        else
            echo "requirements-m1.txt not found."
        fi
    fi
    echo ""
    echo "See also:"
    echo "  • README.md - Full compatibility guide"
    echo "  • SUPPORT_CLI_QUICKSTART.md - Quick start guide"
    echo ""

elif [ "$SYSTEM" = "Darwin" ] && [ "$PLATFORM" = "x86_64" ]; then
    echo -e "${YELLOW}⚠️  Intel Mac or Rosetta detected${NC}"
    echo ""
    if sysctl -n machdep.cpu.brand_string 2>/dev/null | grep -qi "Apple"; then
        echo "Note: You may be running under Rosetta emulation."
        echo "For better performance on M1 Mac, use native ARM Python:"
        echo ""
        echo "  brew install python@3.11"
        echo "  python3 -c 'import platform; print(platform.machine())'  # Should show: arm64"
        echo ""
    fi
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  RECOMMENDED INSTALLATION METHOD: pip"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "Quick start (pip):"
    echo ""
    echo "  ./setup.sh"
    echo ""
    echo "Or manually:"
    echo ""
    echo "  python3 -m venv rag_env"
    echo "  source rag_env/bin/activate"
    echo "  pip install -r requirements.txt"
    echo ""

else
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  RECOMMENDED INSTALLATION METHOD: pip"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "Quick start (pip):"
    echo ""
    echo "  ./setup.sh"
    echo ""
    echo "Or manually:"
    echo ""
    echo "  python3 -m venv rag_env"
    echo "  source rag_env/bin/activate"
    echo "  pip install -r requirements.txt"
    echo ""
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "After installation, build the knowledge base:"
echo ""
echo "  source rag_env/bin/activate  # or: conda activate rag_env"
echo "  python3 -m clockify_rag.cli_modern ingest --input knowledge_helpcenter.md --force"
echo "  python3 -m clockify_rag.cli_modern chat"
echo ""
echo "For more help:"
echo "  • README.md - Main documentation"
echo "  • README.md - Quick start and usage"
echo ""
