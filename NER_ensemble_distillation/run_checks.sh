#! /usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

NER_DIR=$(dirname "$(readlink -f "$0")")
pushd "$NER_DIR" &> /dev/null

# FIXME currently, tests core dump in Gitlab CI's docker (something to do with conflicting TF dependencies)
# Do tests before checking pylint because tests are more important!
echo "[PyTest] Running ner tests"
pytest tests

echo "[Mypy] Checking ner type hints"
mypy . tests scripts

echo "[PyLint] Checking ner module"
find ner -name "*.py" -print0 | xargs -0 pylint --rcfile pylintrc

echo "[PyLint] Checking ner scripts"
find scripts -name "*.py" -print0 | xargs -0 pylint --rcfile pylintrc

echo "[PyLint] Checking ner tests"
# disable redefined-outer-name for tests to use pytest fixtures (e.g., test_divide_sent.py)
# disable protected-access to test protected functions without increasing their visibility (e.g., test_divide_sent.py)
find tests -name "*.py" -print0 | xargs -0 pylint --rcfile pylintrc --disable=C,redefined-outer-name,protected-access

echo "All ner checks passed!"
popd &> /dev/null
