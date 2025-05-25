# Setup Directory

This directory contains setup scripts and configuration files for the project.

## Git Hooks

The `hooks/` directory contains git hooks that help maintain code quality:

- **pre-push**: Runs smoke tests before allowing pushes to prevent broken code from being pushed

### Installing Hooks

Run the install script from the repository root:

```bash
./setup/install-hooks.sh
```

This will:
- Create symlinks from `.git/hooks/` to the hooks in this directory
- Backup any existing hooks (as `.backup` files)
- Allow the hooks to be version controlled and shared with the team

### Bypassing Hooks

In emergency situations, you can bypass hooks with:

```bash
git push --no-verify
```

⚠️ Use this sparingly as it defeats the purpose of having hooks!