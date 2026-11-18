# PySODMetrics Documentation Deployment

This directory contains the source files for building the PySODMetrics documentation.

## Quick Start

Install the project and build the documentation from the repository root.

```bash
python -m pip install -e ".[docs]"
python -m sphinx -b html deploy deploy/_build
```

Open `deploy/_build/index.html` to preview the documentation. GitHub Pages uses
the contents of `deploy/_build/`.

## Clean Build

To rebuild every page with a fresh Sphinx environment, run this from the repository root.

```bash
python -m sphinx -E -a -b html deploy deploy/_build
```

## Directory Structure

- `*.rst` - Documentation source files (reStructuredText)
- `conf.py` - Sphinx configuration
- `_static/` - Optional directory for custom static files (CSS, images, etc.)
