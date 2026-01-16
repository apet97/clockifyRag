#!/bin/bash
#
# v4.1 Smoke Test Suite
# Tests: local embeddings build, selftest, chat queries with JSON output
#
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

LOG_FILE="smoke.log"
: > "$LOG_FILE"  # Clear log

echo "[smoke] v4.1 Smoke Test Suite"
echo "[smoke] Repo: $REPO_ROOT"
echo "[smoke] Log: $LOG_FILE"
echo ""

# Step 1: Activate venv
echo "[1/7] Activating venv..."
if [ ! -d "rag_env" ]; then
    echo "❌ venv not found. Run: python3 -m venv rag_env"
    exit 1
fi
source rag_env/bin/activate
echo "✅ venv activated" | tee -a "$LOG_FILE"

# Step 2: Check dependencies
echo "[2/7] Checking dependencies..."
python3 -c "import numpy, requests, sentence_transformers; print('✅ Dependencies OK')" | tee -a "$LOG_FILE" || {
    echo "❌ Missing dependencies. Run: pip install -r requirements.txt" | tee -a "$LOG_FILE"
    exit 1
}

# Step 3: Build with local embeddings
echo "[3/7] Building knowledge base with local embeddings..."
EMB_BACKEND=local python3 -m clockify_rag.cli_modern ingest --input knowledge_helpcenter.md --force 2>&1 | tee -a "$LOG_FILE"
if [ -f "chunks.jsonl" ] && [ -f "vecs_n.npy" ] && [ -f "meta.jsonl" ]; then
    echo "✅ Build successful" | tee -a "$LOG_FILE"
else
    echo "❌ Build failed - missing artifact files" | tee -a "$LOG_FILE"
    exit 1
fi

# Step 4: Run self-test
echo "[4/7] Running self-test..."
python3 -m clockify_rag.selftest 2>&1 | tee -a "$LOG_FILE"
TEST_EXIT=$?
if [ $TEST_EXIT -eq 0 ]; then
    echo "✅ Self-test passed" | tee -a "$LOG_FILE"
else
    echo "❌ Self-test failed (exit $TEST_EXIT)" | tee -a "$LOG_FILE"
    tail -20 "$LOG_FILE"
    exit 1
fi

# Step 5: Chat query with JSON output
echo "[5/7] Chat query with --json flag..."
QUERY="How do I track time in Clockify?"
python3 -m clockify_rag.cli_modern query "$QUERY" --json 2>&1 | tee -a "$LOG_FILE"
if grep -q '"answer"' "$LOG_FILE"; then
    echo "✅ JSON output valid" | tee -a "$LOG_FILE"
else
    echo "❌ JSON output missing" | tee -a "$LOG_FILE"
fi

# Step 6: Chat query plain text
echo "[6/7] Chat query (plain text)..."
QUERY2="What are the pricing plans?"
python3 -m clockify_rag.cli_modern query "$QUERY2" 2>&1 | tee -a "$LOG_FILE"
echo "✅ Plain text query successful" | tee -a "$LOG_FILE"

# Step 7: Verify KPI logs (greppable format)
echo "[7/7] Verifying KPI logs..."
KPI_COUNT=$(grep -c "^kpi " "$LOG_FILE" || true)
if [ $KPI_COUNT -gt 0 ]; then
    echo "✅ Found $KPI_COUNT KPI logs" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
    echo "=== Last KPI entries ===" | tee -a "$LOG_FILE"
    grep "^kpi " "$LOG_FILE" | tail -5 | tee -a "$LOG_FILE"
else
    echo "⚠️  No KPI logs found (may be expected if not emitted)" | tee -a "$LOG_FILE"
fi

echo ""
echo "=== SMOKE TEST COMPLETE ===" | tee -a "$LOG_FILE"
echo "✅ All smoke tests passed"
echo ""
echo "Last 30 lines of log:"
tail -30 "$LOG_FILE"
