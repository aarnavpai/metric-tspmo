.PHONY: test report lint format

test:
	@uv run pytest suites/benchmark.py \
		--memray -n $(shell nproc --all) \
		--memray-bin-path 'result/memray' \
		--memray-bin-prefix 'memory'

report:
	@uv run script/report.py

type:
	@uv run ty check

lint:
	@uv run ruff check
	@uv run ruff format --check

format:
	@uv run ruff format

charts:
	@uv run script/charts.py
