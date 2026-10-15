#!/usr/bin/env bash
# ==============================================================================
# git-merger: Custom Git Merge Driver Installation Script
#
# Configures Git to use git-merger as a custom merge driver for Python files
# via .gitconfig and .gitattributes.
#
# Usage:
#   ./scripts/install_merge_driver.sh [--global]
# ==============================================================================

set -euo pipefail

GLOBAL_FLAG=""
SCOPE="repository (local)"

if [[ "${1:-}" == "--global" ]]; then
    GLOBAL_FLAG="--global"
    SCOPE="global"
fi

echo "======================================================================"
echo "Installing git-merger semantic merge driver (${SCOPE} scope)..."
echo "======================================================================"

# 1. Configure the custom merge driver command in git config
git config ${GLOBAL_FLAG} merge.git-merger.name "git-merger semantic merge driver"
git config ${GLOBAL_FLAG} merge.git-merger.driver "git-merger resolve --base %O --ours %A --theirs %B --output %A"

echo "✔ Configured merge driver in git config:"
echo "    name:   git-merger semantic merge driver"
echo "    driver: git-merger resolve --base %O --ours %A --theirs %B --output %A"

# 2. Configure .gitattributes (if running in a repository or local mode)
GITATTRIBUTES_FILE=".gitattributes"

if [[ -z "${GLOBAL_FLAG}" ]] || [[ -f "${GITATTRIBUTES_FILE}" ]]; then
    RULE="*.py merge=git-merger"
    if [[ -f "${GITATTRIBUTES_FILE}" ]] && grep -qxF "${RULE}" "${GITATTRIBUTES_FILE}"; then
        echo "✔ ${GITATTRIBUTES_FILE} already contains: ${RULE}"
    else
        echo "${RULE}" >> "${GITATTRIBUTES_FILE}"
        echo "✔ Added '${RULE}' to ${GITATTRIBUTES_FILE}"
    fi
fi

echo ""
echo "======================================================================"
echo "Installation complete!"
echo "Python files (*.py) will now automatically be resolved with git-merger"
echo "whenever a 3-way merge conflict occurs."
echo "======================================================================"
