#!/usr/bin/bash

# move to script directory
cd "$(dirname "$0")"

# run flake8
flake8 openchroma/ --count --show-source --statistics

# run black
black --check --skip-string-normalization openchroma/