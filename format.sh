# move to script directory
cd "$(dirname "$0")"

# run flake8
flake8 . --count --show-source --statistics

# run black
black --check --diff --skip-string-normalization .
