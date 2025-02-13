#!/bin/bash

version=$(git describe --tags --always 2>/dev/null)

# Check if git describe returned a tag (exact match)
if [[ "$version" == *"-"* ]]; then # Check for a dash, indicating commits since a tag
    version=$(git describe --tags --abbrev=0) # Get the closest tag
    version="${version%-*}" # Remove everything after the last dash (including the dash)
    if [[ -z "$version" ]]; then # Check if any tag exists
        version="0.0.0"
    fi
elif [[ -z "$version" ]]; then # No tags at all
    version="0.0.0"
fi

echo "$version" > __version__

sed -i '' "s/version = \"0.0.0\"/version = \"$version\"/" ./pyproject.toml || exit 1