# PySODMetrics Documentation Deployment

This directory contains the source files for building the PySODMetrics documentation.

## Quick Start

Build the documentation:

```bash
cd deploy
sphinx-build -b html . ./_build
```

The built documentation will be placed in the `_build/` directory in the project root.

## Clean Build

To clean the build directory and rebuild from scratch:

```bash
cd deploy
rm -rf ./_build
sphinx-build -b html . ./_build
```

## Directory Structure

- `*.rst` - Documentation source files (reStructuredText)
- `conf.py` - Sphinx configuration
- `_static/` - Static files (CSS, images, etc.)

## Output

- Built HTML documentation is output to: `./deploy/_build/`
- This allows the `deploy/_build/` folder to be used directly for GitHub Pages

## Requirements

Install documentation dependencies:

```bash
pip install sphinx sphinx-rtd-theme
```

Or use the project's optional dependencies:

```bash
pip install -e ".[docs]"
```
