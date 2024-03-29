.PHONY: black-check
black-check:
	poetry run black --check src

.PHONY: black
black:
	poetry run black src

.PHONY: flake8
flake8:
	poetry run flake8 src

.PHONY: isort-check
isort-check:
	poetry run isort --check-only src

.PHONY: isort
isort:
	poetry run isort src

.PHONY: mdformat
mdformat:
	poetry run mdformat *.md

.PHONY: mdformat-check
mdformat-check:
	poetry run mdformat --check *.md

# .PHONY: mypy
# mypy:
# 	poetry run mypy src

# .PHONY: test
# test:
# 	poetry run pytest tests --cov=src --cov-report term-missing --durations 5

.PHONY: format
format:
	$(MAKE) black
	$(MAKE) isort
	$(MAKE) mdformat

.PHONY: lint
lint:
	$(MAKE) black-check
	$(MAKE) isort-check
	$(MAKE) mdformat-check
	$(MAKE) flake8
#	$(MAKE) mypy

# .PHONY: test-all
# test-all:
# 	$(MAKE) lint
# 	$(MAKE) test
