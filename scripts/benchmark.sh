#!/bin/bash
#
# Performance Benchmarking Script
# Measures build and query performance for M1 vs Intel comparison
#
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

LOG_FILE="benchmark_$(date +%Y%m%d_%H%M%S).log"

echo "=== Clockify RAG CLI Performance Benchmark ===" | tee "$LOG_FILE"
echo "Date: $(date)" | tee -a "$LOG_FILE"
echo "Repo: $REPO_ROOT" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# System information
echo "[System Information]" | tee -a "$LOG_FILE"
PLATFORM=$(python3 -c "import platform; print(platform.machine())")
SYSTEM=$(python3 -c "import platform; print(platform.system())")
PYTHON_VERSION=$(python3 --version 2>&1)
PROCESSOR=$(python3 -c "import platform; print(platform.processor())" || echo "N/A")

echo "  System: $SYSTEM" | tee -a "$LOG_FILE"
echo "  Machine: $PLATFORM" | tee -a "$LOG_FILE"
echo "  Python: $PYTHON_VERSION" | tee -a "$LOG_FILE"
echo "  Processor: $PROCESSOR" | tee -a "$LOG_FILE"

# Detect platform type
if [ "$PLATFORM" = "arm64" ] && [ "$SYSTEM" = "Darwin" ]; then
    PLATFORM_TYPE="M1/M2/M3 (Apple Silicon)"
elif [ "$PLATFORM" = "x86_64" ] && [ "$SYSTEM" = "Darwin" ]; then
    PLATFORM_TYPE="Intel Mac"
elif [ "$PLATFORM" = "x86_64" ] && [ "$SYSTEM" = "Linux" ]; then
    PLATFORM_TYPE="Linux x86_64"
else
    PLATFORM_TYPE="$SYSTEM $PLATFORM"
fi

echo "  Platform Type: $PLATFORM_TYPE" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Dependency versions
echo "[Dependency Versions]" | tee -a "$LOG_FILE"
python3 -c "import numpy; print(f'  numpy: {numpy.__version__}')" 2>&1 | tee -a "$LOG_FILE" || echo "  numpy: NOT INSTALLED" | tee -a "$LOG_FILE"
python3 -c "import requests; print(f'  requests: {requests.__version__}')" 2>&1 | tee -a "$LOG_FILE" || echo "  requests: NOT INSTALLED" | tee -a "$LOG_FILE"
python3 -c "import sentence_transformers; print(f'  sentence-transformers: {sentence_transformers.__version__}')" 2>&1 | tee -a "$LOG_FILE" || echo "  sentence-transformers: NOT INSTALLED" | tee -a "$LOG_FILE"
python3 -c "import torch; print(f'  torch: {torch.__version__}')" 2>&1 | tee -a "$LOG_FILE" || echo "  torch: NOT INSTALLED" | tee -a "$LOG_FILE"
python3 -c "import faiss; print(f'  faiss-cpu: {faiss.__version__}')" 2>&1 | tee -a "$LOG_FILE" || echo "  faiss-cpu: NOT INSTALLED" | tee -a "$LOG_FILE"

# PyTorch MPS availability (M1 only)
if [ "$PLATFORM" = "arm64" ] && [ "$SYSTEM" = "Darwin" ]; then
    MPS_AVAILABLE=$(python3 -c "import torch; print(torch.backends.mps.is_available())" 2>&1 || echo "False")
    echo "  PyTorch MPS: $MPS_AVAILABLE" | tee -a "$LOG_FILE"
fi
echo "" | tee -a "$LOG_FILE"

# Check if knowledge base exists
KB_PATH="knowledge_helpcenter.md"
if [ -f "$KB_PATH" ]; then
    KB_LABEL="$KB_PATH"
else
    echo "❌ knowledge_helpcenter.md not found. Cannot run benchmarks." | tee -a "$LOG_FILE"
    exit 1
fi

KB_SIZE=$(du -h "$KB_PATH" | cut -f1)
echo "  Knowledge base: $KB_LABEL" | tee -a "$LOG_FILE"
echo "  Knowledge base size: $KB_SIZE" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Clean old artifacts for fresh build
echo "[Preparation] Cleaning old artifacts..." | tee -a "$LOG_FILE"
rm -f chunks.jsonl vecs_n.npy meta.jsonl bm25.json faiss.index index.meta.json
echo "  ✅ Artifacts cleaned" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Benchmark 1: Build performance
echo "=== Benchmark 1: Build Performance ===" | tee -a "$LOG_FILE"
echo "Building knowledge base..." | tee -a "$LOG_FILE"

BUILD_START=$(date +%s)
python3 -m clockify_rag.cli_modern ingest --input "$KB_PATH" --force 2>&1 | tee -a "$LOG_FILE"
BUILD_END=$(date +%s)
BUILD_TIME=$((BUILD_END - BUILD_START))

echo "" | tee -a "$LOG_FILE"
echo "  ⏱️  Build time: ${BUILD_TIME}s" | tee -a "$LOG_FILE"

# Check artifacts
if [ -f "chunks.jsonl" ] && [ -f "vecs_n.npy" ] && [ -f "meta.jsonl" ]; then
    CHUNKS_COUNT=$(wc -l < chunks.jsonl)
    CHUNKS_SIZE=$(du -h chunks.jsonl | cut -f1)
    VECS_SIZE=$(du -h vecs_n.npy | cut -f1)

    echo "  Artifacts:" | tee -a "$LOG_FILE"
    echo "    - chunks.jsonl: $CHUNKS_COUNT chunks ($CHUNKS_SIZE)" | tee -a "$LOG_FILE"
    echo "    - vecs_n.npy: $VECS_SIZE" | tee -a "$LOG_FILE"

    if [ -f "faiss.index" ]; then
        FAISS_SIZE=$(du -h faiss.index | cut -f1)
        echo "    - faiss.index: $FAISS_SIZE" | tee -a "$LOG_FILE"
    else
        echo "    - faiss.index: not created (fallback mode)" | tee -a "$LOG_FILE"
    fi
else
    echo "  ❌ Build failed - missing artifacts" | tee -a "$LOG_FILE"
    exit 1
fi
echo "" | tee -a "$LOG_FILE"

# Benchmark 2: Query performance (single query)
echo "=== Benchmark 2: Query Performance ===" | tee -a "$LOG_FILE"

# Test queries (various complexities)
QUERIES=(
    "How do I track time in Clockify?"
    "What are the pricing plans?"
    "Can I track time offline?"
)

TOTAL_QUERY_TIME=0
QUERY_COUNT=0

for QUERY in "${QUERIES[@]}"; do
    echo "Query $((QUERY_COUNT+1)): \"$QUERY\"" | tee -a "$LOG_FILE"

    QUERY_START=$(date +%s%3N)  # milliseconds
    python3 -m clockify_rag.cli_modern query "$QUERY" > /tmp/query_result.txt 2>&1 || echo "Query failed" > /tmp/query_result.txt
    QUERY_END=$(date +%s%3N)
    QUERY_TIME=$((QUERY_END - QUERY_START))

    echo "  ⏱️  Query time: ${QUERY_TIME}ms" | tee -a "$LOG_FILE"

    # Extract answer length
    ANSWER_LENGTH=$(wc -c < /tmp/query_result.txt)
    echo "  Answer length: $ANSWER_LENGTH chars" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"

    TOTAL_QUERY_TIME=$((TOTAL_QUERY_TIME + QUERY_TIME))
    QUERY_COUNT=$((QUERY_COUNT + 1))
done

AVG_QUERY_TIME=$((TOTAL_QUERY_TIME / QUERY_COUNT))
echo "  Average query time: ${AVG_QUERY_TIME}ms" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Benchmark 3: Memory usage (if available)
echo "=== Benchmark 3: Memory Usage ===" | tee -a "$LOG_FILE"

if command -v ps &> /dev/null; then
    # Start a background process and measure memory
    python3 -m clockify_rag.cli_modern query "Test query" > /dev/null 2>&1 &
    PID=$!
    sleep 2

    if ps -p $PID > /dev/null 2>&1; then
        if [ "$SYSTEM" = "Darwin" ]; then
            MEM_KB=$(ps -o rss= -p $PID 2>/dev/null || echo "0")
            MEM_MB=$((MEM_KB / 1024))
        else
            MEM_KB=$(ps -o rss= -p $PID 2>/dev/null || echo "0")
            MEM_MB=$((MEM_KB / 1024))
        fi

        echo "  Process memory (peak): ${MEM_MB} MB" | tee -a "$LOG_FILE"
    else
        echo "  ⚠️  Could not measure memory (process exited)" | tee -a "$LOG_FILE"
    fi

    # Kill background process if still running
    kill $PID 2>/dev/null || true
    wait $PID 2>/dev/null || true
else
    echo "  ⚠️  ps command not available, skipping memory measurement" | tee -a "$LOG_FILE"
fi
echo "" | tee -a "$LOG_FILE"

# Summary
echo "=== Performance Summary ===" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Platform: $PLATFORM_TYPE" | tee -a "$LOG_FILE"
echo "Python: $PYTHON_VERSION" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Build Performance:" | tee -a "$LOG_FILE"
echo "  Total time: ${BUILD_TIME}s" | tee -a "$LOG_FILE"
echo "  Chunks created: $CHUNKS_COUNT" | tee -a "$LOG_FILE"
echo "  Throughput: $((CHUNKS_COUNT / BUILD_TIME)) chunks/sec" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Query Performance:" | tee -a "$LOG_FILE"
echo "  Average latency: ${AVG_QUERY_TIME}ms" | tee -a "$LOG_FILE"
echo "  Queries tested: $QUERY_COUNT" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Platform-specific notes
if [ "$PLATFORM" = "arm64" ] && [ "$SYSTEM" = "Darwin" ]; then
    echo "M1/M2/M3 Optimization Status:" | tee -a "$LOG_FILE"

    # Check if ARM64 optimization was used
    if grep -q "macOS arm64 detected" "$LOG_FILE"; then
        echo "  ✅ ARM64 optimizations activated (FlatIP)" | tee -a "$LOG_FILE"
    else
        echo "  ⚠️  ARM64 optimization message not detected" | tee -a "$LOG_FILE"
    fi

    if [ "$MPS_AVAILABLE" = "True" ]; then
        echo "  ✅ PyTorch MPS available (GPU acceleration)" | tee -a "$LOG_FILE"
    else
        echo "  ⚠️  PyTorch MPS not available (CPU only)" | tee -a "$LOG_FILE"
    fi

    echo "" | tee -a "$LOG_FILE"
    echo "Expected Performance (M1 Pro, 16GB):" | tee -a "$LOG_FILE"
    echo "  Build time: ~30s" | tee -a "$LOG_FILE"
    echo "  Query latency: ~6-11s" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
fi

echo "=== Benchmark Complete ===" | tee -a "$LOG_FILE"
echo "Results saved to: $LOG_FILE" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Export CSV for comparison
CSV_FILE="benchmark_results.csv"
if [ ! -f "$CSV_FILE" ]; then
    echo "timestamp,platform,system,machine,python_version,build_time_s,avg_query_ms,chunks_count,faiss_available,mps_available" > "$CSV_FILE"
fi

FAISS_AVAIL=$([ -f "faiss.index" ] && echo "yes" || echo "no")
MPS_AVAIL="${MPS_AVAILABLE:-N/A}"
TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")

echo "\"$TIMESTAMP\",\"$PLATFORM_TYPE\",\"$SYSTEM\",\"$PLATFORM\",\"$PYTHON_VERSION\",$BUILD_TIME,$AVG_QUERY_TIME,$CHUNKS_COUNT,$FAISS_AVAIL,$MPS_AVAIL" >> "$CSV_FILE"

echo "CSV results appended to: $CSV_FILE"
echo "Use this file to compare performance across different platforms."
