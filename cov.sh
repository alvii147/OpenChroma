#!/usr/bin/bash

# move to script parent directory
cd "$(dirname "$0")"

# set PYTHONPATH
export PYTHONPATH=${PWD}

# run all tests
coverage run -m pytest -v -s tests/

# get coverage report
coverage report

# erase coverage
coverage erase