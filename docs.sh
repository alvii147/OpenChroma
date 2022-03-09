# move to docs directory
cd "$(dirname "$0")/docs/"

# remove current html docs
rm -rf build/

# build html docs
make html
