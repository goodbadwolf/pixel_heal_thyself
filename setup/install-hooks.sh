#!/bin/bash

# Install git hooks for the project

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Get the repository root
REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null)

if [ -z "$REPO_ROOT" ]; then
    echo "Error: Not in a git repository"
    exit 1
fi

# Source and destination directories
HOOKS_SOURCE="$REPO_ROOT/setup/hooks"
GIT_HOOKS_DIR="$REPO_ROOT/.git/hooks"

echo "Installing git hooks..."

# Install each hook
for hook in "$HOOKS_SOURCE"/*; do
    if [ -f "$hook" ]; then
        hook_name=$(basename "$hook")
        dest="$GIT_HOOKS_DIR/$hook_name"
        
        # Backup existing hook if it exists
        if [ -f "$dest" ] && [ ! -L "$dest" ]; then
            echo -e "${YELLOW}Backing up existing $hook_name to $hook_name.backup${NC}"
            mv "$dest" "$dest.backup"
        fi
        
        # Create symlink
        ln -sf "$hook" "$dest"
        echo -e "${GREEN}✓ Installed $hook_name${NC}"
    fi
done

echo -e "${GREEN}Git hooks installation complete!${NC}"
echo -e "${YELLOW}Note: You can bypass hooks with 'git push --no-verify' if needed${NC}"