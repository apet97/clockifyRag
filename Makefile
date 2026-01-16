.PHONY: help venv install deps-check selftest build chat smoke smoke-full clean dev test eval benchmark benchmark-quick typecheck lint format pre-commit-install pre-commit-run regen-artifacts rebuild-all test-quick verify verify-ollama eval-gate

help:
	@echo "v4.1 Clockify RAG CLI - Make Targets"
	@echo ""
	@echo "  make dev                 - Setup development environment (venv + install + pre-commit)"
	@echo "  make venv                - Create Python virtual environment"
	@echo "  make install             - Install dependencies via pyproject (requires venv)"
	@echo "  make deps-check          - Run pip check + targeted pytest smoke"
	@echo "  make build               - Build knowledge base (uses local embeddings for speed)"
	@echo "  make ingest              - Alias for 'make build' (ragctl ingest)"
	@echo "  make reindex             - Force rebuild of KB/index artifacts"
	@echo "  make regen-artifacts     - Regenerate derived artifacts (chunk_title_map.json, etc.)"
	@echo "  make rebuild-all         - Clean and rebuild all artifacts from scratch"
	@echo "  make selftest            - Run self-test suite"
	@echo "  make chat                - Start interactive chat (REPL)"
	@echo "  make smoke               - Run full smoke test suite"
	@echo "  make test-quick          - Fast unit tests (config/API client/answer core)"
	@echo "  make verify              - pip check + make test-quick + make smoke + make eval-gate"
	@echo "  make verify-ollama       - Run verify with SMOKE_CLIENT=ollama (VPN required)"
	@echo "  make test                - Run unit tests with coverage"
	@echo "  make eval                - Run RAG evaluation on ground truth dataset"
	@echo "  make benchmark           - Run performance benchmarks (latency, throughput, memory)"
	@echo "  make benchmark-quick     - Run quick benchmarks (fewer iterations)"
	@echo "  make typecheck           - Run mypy static type checking"
	@echo "  make lint                - Run ruff linter"
	@echo "  make format              - Format code with black"
	@echo "  make pre-commit-install  - Install pre-commit git hooks"
	@echo "  make pre-commit-run      - Run pre-commit hooks on all files"
	@echo "  make clean               - Remove generated artifacts and cache"
	@echo ""
	@echo "Quick start:"
	@echo "  make dev  (then follow on-screen instructions)"
	@echo ""

venv:
	@echo "Creating virtual environment..."
	python3 -m venv rag_env
	@echo "✅ venv created. Activate: source rag_env/bin/activate"

install:
	@echo "Installing dependencies..."
	@if [ -f requirements.lock ]; then \
		echo "Installing from lockfile..."; \
		source rag_env/bin/activate && pip install -q -r requirements.lock; \
	else \
		echo "Installing editable package with dev extras..."; \
		source rag_env/bin/activate && pip install -q -e '.[dev]'; \
	fi
	@echo "✅ Dependencies installed"

.PHONY: deps-check
deps-check:
	@echo "Running dependency health checks..."
	@if [ ! -d rag_env ]; then \
		echo "❌ rag_env not found. Run 'make dev' first."; \
		exit 1; \
	fi
	source rag_env/bin/activate && python -m pip check
	source rag_env/bin/activate && python -m pytest tests/test_api_client.py tests/test_config_module.py
	@echo "✅ Dependency check suite passed"

.PHONY: test-quick
test-quick:
	@echo "Running quick verification tests..."
	python3 -m pytest tests/test_config_module.py tests/test_api_client.py tests/test_answer.py tests/test_cli_dependencies.py

.PHONY: freeze
freeze:
	@echo "Generating requirements.lock..."
	source rag_env/bin/activate && pip freeze > requirements.lock
	@echo "✅ Lockfile generated"

build:
	@echo "Building knowledge base with ragctl ingest (local embeddings for speed)..."
	source rag_env/bin/activate && EMB_BACKEND=local python3 -m clockify_rag.cli_modern ingest --input knowledge_helpcenter.md
	@echo ""
	@echo "Hint: To use Ollama embeddings instead, run: EMB_BACKEND=ollama make build"

ingest: build
	@true

reindex:
	@echo "Force rebuilding knowledge base and indexes..."
	source rag_env/bin/activate && EMB_BACKEND=local python3 -m clockify_rag.cli_modern ingest --input knowledge_helpcenter.md --force
	@echo "✅ Reindex complete"

regen-artifacts:
	@echo "Regenerating derived artifacts..."
	@if [ ! -f chunks.jsonl ]; then \
		echo "Error: chunks.jsonl not found. Run 'make build' first."; \
		exit 1; \
	fi
	@echo "Regenerating chunk_title_map.json..."
	source rag_env/bin/activate && python3 scripts/generate_chunk_title_map.py
	@echo "✅ Artifacts regenerated"
	@echo ""
	@echo "Note: Run this after rebuilding the knowledge base to keep chunk_title_map.json in sync"

rebuild-all:
	@echo "Rebuilding all artifacts from scratch..."
	@echo ""
	@echo "Step 1/3: Cleaning old artifacts..."
	$(MAKE) clean
	@echo ""
	@echo "Step 2/3: Building knowledge base..."
	$(MAKE) build
	@echo ""
	@echo "Step 3/3: Generating derived artifacts..."
	$(MAKE) regen-artifacts
	@echo ""
	@echo "✅ Full rebuild complete!"
	@echo ""
	@echo "All artifacts generated:"
	@echo "  - chunks.jsonl (text chunks)"
	@echo "  - vecs_n.npy (normalized embeddings)"
	@echo "  - meta.jsonl (chunk metadata)"
	@echo "  - bm25.json (BM25 index)"
	@echo "  - faiss.index (FAISS ANN index)"
	@echo "  - index.meta.json (version metadata)"
	@echo "  - chunk_title_map.json (ID→title mapping)"
	@echo ""
	@echo "Use 'make chat' to start querying!"

selftest:
	@echo "Running self-test suite..."
	source rag_env/bin/activate && python3 -m clockify_rag.selftest

chat:
	@echo "Starting interactive chat (REPL)..."
	source rag_env/bin/activate && python3 -m clockify_rag.cli_modern chat

SMOKE_CLIENT ?= mock
EVAL_DATASET ?= eval_datasets/clockify_v1.jsonl

.PHONY: smoke
smoke:
	@echo "Running lightweight smoke test (SMOKE_CLIENT=$(SMOKE_CLIENT))..."
	python3 scripts/smoke_rag.py --client $(SMOKE_CLIENT) $(SMOKE_ARGS)

.PHONY: verify
verify:
	@echo "Running verification gate (pip check + test-quick + smoke)..."
	@if [ ! -d rag_env ]; then \
		echo "❌ rag_env not found. Run 'make dev' first."; \
		exit 1; \
	fi
	source rag_env/bin/activate && python -m pip check
	$(MAKE) test-quick
	$(MAKE) smoke
	$(MAKE) eval-gate

.PHONY: smoke-full
smoke-full:
	@echo "Running legacy shell smoke harness..."
	bash scripts/smoke.sh

test:
	@echo "Running unit tests with coverage..."
	python3 -m pytest tests/ -v --cov=clockify_rag --cov-report=term-missing --cov-report=html

eval:
	@echo "Running RAG evaluation on ground truth dataset..."
	python3 eval.py --dataset eval_datasets/clockify_v1.jsonl

benchmark:
	@echo "Running performance benchmarks..."
	python3 benchmark.py

benchmark-quick:
	@echo "Running quick benchmarks..."
	python3 benchmark.py --quick

typecheck:
	@echo "Running mypy type checking..."
	python3 -m mypy clockify_rag --config-file pyproject.toml

lint:
	@echo "Running ruff linter..."
	python3 -m ruff check clockify_rag --config pyproject.toml

format:
	@echo "Formatting code with black..."
	python3 -m black clockify_rag --config pyproject.toml

pre-commit-install:
	@echo "Installing pre-commit hooks..."
	python3 -m pre_commit install
	@echo "✅ Pre-commit hooks installed"

pre-commit-run:
	@echo "Running pre-commit hooks on all files..."
	python3 -m pre_commit run --all-files

clean:
	@echo "Cleaning generated artifacts..."
	rm -f chunks.jsonl vecs_n.npy vecs.npy meta.jsonl bm25.json index.meta.json
	rm -f faiss.index hnsw_cosine.bin emb_cache.jsonl chunk_title_map.json
	rm -f .build.lock .shim.pid shim.log build.log smoke.log query.log audit.jsonl
	rm -rf .mypy_cache .pytest_cache htmlcov .ruff_cache
	@echo "✅ Clean complete"

dev: venv install pre-commit-install
	@echo ""
	@echo "✅ Development environment ready!"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Activate venv: source rag_env/bin/activate"
	@echo "  2. Build index: make build"
	@echo "  3. Start chat: make chat"
	@echo ""
	@echo "Or run all steps: source rag_env/bin/activate && make build && make chat"
	@echo ""
.PHONY: verify-ollama
verify-ollama:
	@echo "Running verification gate against real Ollama (VPN required)..."
	SMOKE_CLIENT=ollama $(MAKE) verify

.PHONY: eval-gate
eval-gate:
	@echo "Running retrieval evaluation..."
	@if [ ! -f "$(EVAL_DATASET)" ]; then \
		echo "❌ Evaluation dataset $(EVAL_DATASET) not found. Set EVAL_DATASET=/path/to/dataset.jsonl"; \
		exit 1; \
	fi
	python3 eval.py --dataset "$(EVAL_DATASET)"
