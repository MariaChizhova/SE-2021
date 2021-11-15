#!/usr/bin/env bash
set -e

help=$(cat << EOF
Launch tests, linters or coverage for Hermes

Linters:
./tests.sh linters  # launch all linters listed below
./tests.sh pylint
./tests.sh flake8
./tests.sh mypy

Tests:
./tests.sh test

Coverage:
./tests.sh coverage    # coverage with no acceptance rate
./tests.sh coverage95  # coverage with 95 acceptance rate
./tests.sh coverage99  # coverage with 99 acceptance rate

All:
./tests.sh all

You can mix any of them
EOF
)

if [ "$#" -eq 0 ]; then
    echo "$help"
    exit 0
fi

l_pylint=0
l_flake8=0
l_mypy=0
l_test=0
l_coverage=0

while test $# -gt 0
do
    case "$1" in
        linters) 
            l_pylint=1
            l_flake8=1
            l_mypy=1
            ;;
        pylint) l_pylint=1
            ;;
        flake8) l_flake8=1
            ;;
        mypy) l_mypy=1
            ;;
        test) l_test=1
            ;;
        coverage*) 
            if ! [[ "$1" =~ ^coverage([0-9]*)$ ]]; then
                echo "Not a number or empty string: $1" &>2 && exit 1
            fi
            rate="${BASH_REMATCH[1]}"
            l_coverage=1
            ;;
        all) 
            l_pylint=1
            l_flake8=1
            l_mypy=1
            l_test=1
            l_coverage=1
            ;;
        *) echo "Wrong argument: $1" &>2 && exit 1
            ;;
    esac
    shift
done

if [[ $l_pylint == 1 ]]; then
    python -m pylint --max-line-length=120 --disable=invalid-name,missing-docstring,global-statement,too-many-lines,R,no-member --enable=simplifiable-if-statement,redefined-variable-type hermes test
fi

if [[ $l_flake8 == 1 ]]; then
python -m flake8 --max-line-length=120 hermes test
fi

if [[ $l_mypy == 1 ]]; then
python -m mypy --ignore-missing-imports hermes test
fi

if [[ $l_test == 1 && $l_coverage == 0 ]]; then
    python -m pytest test
elif [[ $l_coverage == 1 ]]; then
    coverage run -m pytest test
    if [[ $rate != "" ]]; then
        coverage report --fail-under "$rate"
    fi
fi