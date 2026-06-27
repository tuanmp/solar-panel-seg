UV := uv

.PHONY: sync train test lint format train_solar

sync:
	$(UV) sync --group dev

train:
	$(UV) run python -m solar_seg.train

train_solar:
	$(UV) run python -m solar_seg.train

test:
	$(UV) run pytest

lint:
	$(UV) run python -m compileall src tests

format:
	@echo "No formatter configured yet"
