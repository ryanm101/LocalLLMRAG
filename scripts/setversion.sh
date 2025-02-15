#!/bin/bash -xe

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
version=$(git describe --tags --always 2>/dev/null)

# Normalize version to be PEP 440-compliant
if [[ "$version" == *"-"* ]]; then
    base_version=$(git describe --tags --abbrev=0 2>/dev/null | sed 's/^v//') # Remove leading "v" if present
    commits_since_tag=$(git rev-list --count HEAD ^$base_version 2>/dev/null || echo 0)
    version="${base_version}.post${commits_since_tag}"
    if [[ -z "$base_version" ]]; then
        version="0.0.0"
    fi
elif [[ -z "$version" ]]; then
    version="0.0.0"
else
    version="${version#v}"  # Remove leading "v" if present
fi

# Ensure version is valid PEP 440
if ! [[ "$version" =~ ^[0-9]+(\.[0-9]+)*(\.post[0-9]+)?$ ]]; then
    echo "Error: Generated version '$version' is not PEP 440-compliant"
    exit 1
fi

echo "$version" > __version__

# Use cross-platform sed for macOS & Linux
if [[ "$OSTYPE" == "darwin"* ]]; then
    sed -i '' "s/version = \"0.0.0\"/version = \"$version\"/" "${SCRIPT_DIR}/../pyproject.toml"
else
    sed -i "s/version = \"0.0.0\"/version = \"$version\"/" "${SCRIPT_DIR}/../pyproject.toml"
fi