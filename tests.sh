#!/usr/bin/env bash

set -e

python -m pylint hermes test
python -m flake8 hermes test
python -m mypy hermes test
coverage run -m pytest test
coverage report --fail-under 95