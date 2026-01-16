# CR_Score Documentation

This directory contains the Sphinx documentation for CR_Score.

## Building Documentation

### Prerequisites

```bash
pip install -r requirements.txt
```

### Build HTML Documentation

**Linux/Mac:**
```bash
cd docs
make html
```

**Windows:**
```bash
cd docs
make.bat html
```

### View Documentation

Open `docs/build/html/index.html` in your browser.

### Build PDF Documentation

**Linux/Mac:**
```bash
cd docs
make latexpdf
```

### Clean Build

```bash
cd docs
make clean
```

## Documentation Structure

```
docs/
├── source/
│   ├── index.rst              # Main documentation index
│   ├── conf.py                # Sphinx configuration
│   ├── api/                   # API reference documentation
│   ├── guides/                # User guides
│   ├── examples/              # Example code documentation
│   ├── _static/               # Static files (images, CSS)
│   └── _templates/            # Custom templates
├── build/                     # Generated documentation (gitignored)
├── Makefile                   # Linux/Mac build script
├── make.bat                   # Windows build script
└── requirements.txt           # Documentation build requirements
```

## Documentation Guidelines

- Use Google-style docstrings in code
- Keep examples simple and focused
- Include code samples in guides
- Cross-reference related documentation
- Update changelog for significant changes

## ReadTheDocs

Documentation is automatically built and deployed to ReadTheDocs on every commit to main.

Visit: https://cr-score.readthedocs.io (or your configured URL)

## Local Development

For live reload during documentation development:

```bash
pip install sphinx-autobuild
cd docs
sphinx-autobuild source build/html
```

Then open http://localhost:8000 in your browser.
