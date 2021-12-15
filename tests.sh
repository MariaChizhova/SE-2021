#!/usr/bin/env bash

set -e

python -m pylint --max-line-length=120 --disable=invalid-name,missing-docstring,global-statement,too-many-lines,R,no-member --enable=simplifiable-if-statement,redefined-variable-type hermes test
python -m flake8 --max-line-length=120 hermes test
python -m mypy --ignore-missing-imports hermes test
coverage run -m pytest test
coverage report --fail-under 95