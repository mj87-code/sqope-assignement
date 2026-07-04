# Load .env (ANTHROPIC_API_KEY, DB_PASSWORD required; HF_TOKEN optional) if present.
-include .env
export

PDF ?= $(PWD)/data/data_test_novatech_q4_2024_report_1.pdf
DB_PASSWORD ?= secret
# Optional: pass TRACE=1 to `make ask` to print each answer's internal
# pipeline steps. Left off by default.
TRACE ?=
# Optional: only silences a HuggingFace rate-limit warning during model
# downloads. Everything works fine with this left unset.
HF_TOKEN ?=

.PHONY: lint format test test-integration test-all eval docker-up docker-index docker-test docker-down ask db-reset docker-reset fresh

# Lint with ruff (style, imports, pyflakes, bugbear, pyupgrade)
lint:
	uv run ruff check .

# Auto-fix lint issues and format
format:
	uv run ruff check . --fix
	uv run ruff format .

# Fast unit tests — no DB, no LLM needed
test:
	uv run pytest tests/ -v -k "not integration"

# Integration tests — requires API running at API_URL (default: http://localhost:8000)
test-integration:
	uv run pytest tests/test_api.py -v -m integration

# All tests (unit + integration)
test-all:
	uv run pytest tests/ -v

# Quantified eval suite (intent accuracy, retrieval recall@k/MRR, e2e answer
# correctness, faithfulness precision/recall/F2) — real calls against the live
# pipeline. Requires ANTHROPIC_API_KEY + the DB reachable on localhost:5432
# (e.g. via `make docker-up`) + the API running.
eval:
	DATABASE_URL=postgresql://sqope:$(DB_PASSWORD)@localhost:5432/sqope \
	TEST_DATABASE_URL=postgresql://sqope:$(DB_PASSWORD)@localhost:5432/sqope \
	  uv run python -m evals.eval_suite

# --- Docker workflow (no local Python needed) ---

# Start DB + API, and build the indexer image (needed by docker-index/ask,
# but never started as a persistent service — it's a one-shot container).
# Pass LOGS=1 to follow the API logs after starting: make docker-up LOGS=1
docker-up:
	DB_PASSWORD=$(DB_PASSWORD) docker-compose up --build -d db api
	DB_PASSWORD=$(DB_PASSWORD) docker-compose build indexer
ifdef LOGS
	docker-compose logs -f api
endif

# Index a PDF (one-shot container). Override the file: make docker-index PDF=/path/to/your.pdf
docker-index:
	docker run --rm \
	  --network sqope-ai-home-assignment-task_default \
	  -e DATABASE_URL=postgresql://sqope:$(DB_PASSWORD)@db:5432/sqope \
	  -e HF_TOKEN=$(HF_TOKEN) \
	  -v "$(PDF):/data/report.pdf" \
	  sqope-ai-home-assignment-task-indexer \
	  python -m indexer.cli /data/report.pdf

# Interactive chat client — ask the running API questions in your terminal.
# Pass TRACE=1 to also print each answer's internal pipeline steps: make ask TRACE=1
ask:
	docker run -it --rm \
	  --network sqope-ai-home-assignment-task_default \
	  -e API_URL=http://api:8000 \
	  -e SHOW_TRACE=$(TRACE) \
	  sqope-ai-home-assignment-task-indexer \
	  python ask.py

# Run integration tests inside Docker (no local Python required)
docker-test:
	DB_PASSWORD=$(DB_PASSWORD) docker-compose --profile test run --rm test

# Stop all containers (keeps the DB volume / indexed data)
docker-down:
	docker-compose down

# Clear indexed documents but keep the stack running (fast data reset)
db-reset:
	docker exec sqope-ai-home-assignment-task-db-1 psql -U sqope -d sqope \
	  -c "TRUNCATE documents, text_chunks, table_rows CASCADE;"

# Full wipe: stop containers AND delete the DB volume (schema recreated on next up)
docker-reset:
	docker-compose down -v

# Start completely fresh: wipe everything, then build + start DB + API.
# Pass LOGS=1 to follow the API logs after starting: make fresh LOGS=1
fresh: docker-reset
	DB_PASSWORD=$(DB_PASSWORD) docker-compose up --build -d db api
	DB_PASSWORD=$(DB_PASSWORD) docker-compose build indexer
	@echo "Stack is fresh and empty. Index a PDF with: make docker-index PDF=/path/to.pdf"
ifdef LOGS
	docker-compose logs -f api
endif
