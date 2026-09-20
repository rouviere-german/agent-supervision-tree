# Linux/CI convenience targets. Every target delegates to tasks.py, which is the single source
# of truth and also runs on Windows: `uv run python tasks.py <target>`.

UV ?= uv
TASK := $(UV) run python tasks.py
IMAGE ?= agent-supervision-tree:local

.PHONY: install lint format typecheck test bench soak demo bandit pip-audit audit baseline record check docker-build compose-up

install:
	$(UV) sync --frozen

lint:
	$(TASK) lint

format:
	$(TASK) format

typecheck:
	$(TASK) typecheck

test:
	$(TASK) test

bench:
	$(TASK) bench

soak:
	$(TASK) soak

demo:
	$(TASK) demo

bandit:
	$(TASK) bandit

pip-audit:
	$(TASK) pip-audit

audit:
	$(TASK) audit

baseline:
	$(TASK) baseline

record:
	$(TASK) record

check:
	$(TASK) check

docker-build:
	docker build -t $(IMAGE) .

compose-up:
	docker compose up --build
