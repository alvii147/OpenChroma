#!/usr/bin/bash

# move to script parent directory
cd "$(dirname "$0")"

# set PYTHONPATH
export PYTHONPATH=${PWD}

# run all tests
coverage run --source openchroma -m pytest -v -s tests/

# get coverage report
coverage report -m

# erase coverage
coverage erase