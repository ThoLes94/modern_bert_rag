PACKAGES=src
STYLE_PACKAGES=$(PACKAGES)
TEST_PACKAGES=$(PACKAGES) test


check_black:
	# python -m ruff format --check --diff ${STYLE_PACKAGES}
	python -m black -t py310 --line-length 100 --check --diff ${STYLE_PACKAGES}

black:
	python -m black -t py310 --line-length 100 ${STYLE_PACKAGES}

isort:
	ruff check --select I --fix ${STYLE_PACKAGES}

check_isort:
	ruff check --select I ${STYLE_PACKAGES}

style: black isort

checks: style lint mypy

checkstyle: check_black check_isort
check_style: checkstyle

lint:
	python -m flake8 ${STYLE_PACKAGES}

pre-commit-install:
	pre-commit install -c .pre-commit-config.yaml

pre-commit:
	pre-commit run --all-files -c .pre-commit-config.yaml

mypy:
	mypy src

launch_backend:
	uvicorn src.backend.api.main:app --reload --reload-dir src/backend

launch_frontend:
	chainlit run src/frontend/my_cl_app.py --port 8501
