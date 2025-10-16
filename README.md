<h1 align="center">Automate dir zipping⚡️</h1>
CLI for flexibly zipping and extracting items.

## Features
* compress files and dirs as required
    - run command for zipping dirs/files of choice
    - run command for defining where compressed files should live

## Prerequisites
* Python 3.12+
* Git
* uv (recommended) or pip

## Quick Start

```bash
# Compress current directory
$ uv run zippa pack .

# Compress specific files
$ uv run zippa pack file1.txt dir1/ file2.py

# Extract zip file
$ uv run zippa extract backup.zip

# Extract to specific directory
$ uv run zippa extract backup.zip --output /path/to/extract/
```

## Common Patterns

### Working with Different Directories

```bash
# Compress from external directory
$ uv run zippa pack . --work-dir ~/path/to/project --output backup.zip

# Compress specific files from external directory
$ uv run zippa pack file1.txt dir1/ --work-dir ~/path/to/project
```

### Excluding Files

```bash
# Exclude patterns via command line
$ uv run zippa pack . --exclude "*.pyc" --exclude "__pycache__"

# Use .zipignore file (create in project root)
$ uv run zippa pack . --exclude-file .zipignore
```

### Overwrite Protection

```bash
# Default: Skip if files exist (safe)
$ uv run zippa pack . --output backup.zip

# Ask before overwriting (interactive)
$ uv run zippa pack . --output backup.zip --ask

# Force overwrite (non-interactive)
$ uv run zippa pack . --output backup.zip --force-overwrite
```

## .zipignore File

Create a `.zipignore` file in your project root:

```
# .zipignore example
*.pyc
__pycache__/
*.log
*.tmp
.git/
node_modules/
.env
*.DS_Store
build/
dist/
*.egg-info/
```

## Notes

- Items are relative to your current working directory (or `--work-dir` if specified)
- Output location can be anywhere (absolute or relative path)
- Use `--help` for complete command reference


## Future Updates
* [ ] make syntax for running commands more intuitive
* [ ] implement compression summary after each compression
* [ ] listen to events and compress correspondently
