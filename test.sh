# move to script parent directory
cd "$(dirname "$0")"

# set PYTHONPATH
export PYTHONPATH=${PWD}

# run all tests
pytest -v -s tests/
