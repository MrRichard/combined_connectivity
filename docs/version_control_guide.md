# Version Control Guide

This guide explains how to manage version control for the combined connectivity pipelines, which consist of three separate components that can be maintained independently.

## Repository Structure

The project is designed as **three independent repositories** that can be deployed separately:

```
combined_connectivity/           # Parent directory (optional wrapper repo)
├── connectivity_shared/         # Shared utilities package
├── mrtrix3_demon_addon/         # Diffusion pipeline
└── nilearn_RSB_analysis_pipeline/  # fMRI pipeline
```

### Option 1: Monorepo (Simple)

Keep everything in a single repository. Best for:
- Small teams
- Tightly coupled development
- Simpler CI/CD

```bash
# Initialize as single repo
cd combined_connectivity
git init
git add .
git commit -m "Initial commit: combined connectivity pipelines"
```

### Option 2: Separate Repositories (Recommended)

Maintain each component as its own repository. Best for:
- Independent deployment
- Different release cycles
- Multiple teams

```bash
# Initialize each component separately
cd connectivity_shared
git init
git add .
git commit -m "Initial commit: connectivity-shared package"

cd ../mrtrix3_demon_addon
git init
git add .
git commit -m "Initial commit: MRtrix3 diffusion pipeline"

cd ../nilearn_RSB_analysis_pipeline
git init
git add .
git commit -m "Initial commit: nilearn fMRI pipeline"
```

## Dependency Management

### Installing connectivity_shared

The pipelines depend on `connectivity_shared`. There are several ways to manage this:

#### Development Mode (Local Development)

```bash
# Install from local path (editable)
pip install -e /path/to/connectivity_shared
```

#### Requirements File

Add to each pipeline's `requirements.txt`:

```
# For local development
-e /path/to/connectivity_shared

# OR for production (if published to PyPI)
connectivity-shared>=0.1.0

# OR for git-based installation
git+https://github.com/yourorg/connectivity-shared.git@v0.1.0
```

#### Conda Environment

Create an environment file (`environment.yml`):

```yaml
name: connectivity
channels:
  - conda-forge
dependencies:
  - python>=3.8
  - numpy
  - pandas
  - scipy
  - networkx
  - nibabel
  - matplotlib
  - pip
  - pip:
    - -e /path/to/connectivity_shared
```

## Branching Strategy

### Recommended: Git Flow

```
main          # Production-ready code
├── develop   # Integration branch
├── feature/* # New features
├── bugfix/*  # Bug fixes
└── release/* # Release preparation
```

### Branch Naming Conventions

```bash
# Features
git checkout -b feature/add-destrieux-atlas

# Bug fixes
git checkout -b bugfix/fix-matrix-loading

# Releases
git checkout -b release/v0.2.0
```

## Versioning

Use [Semantic Versioning](https://semver.org/):

```
MAJOR.MINOR.PATCH

0.1.0  # Initial development
0.2.0  # New features (backward compatible)
0.2.1  # Bug fixes
1.0.0  # First stable release
```

### Updating Versions

**connectivity_shared** (`pyproject.toml`):
```toml
[project]
version = "0.1.0"
```

**Pipeline scripts** (update `__version__` in main scripts):
```python
__version__ = "0.1.0"
```

## Synchronizing Changes

### When connectivity_shared Changes

1. **Update the shared package:**
   ```bash
   cd connectivity_shared
   git add .
   git commit -m "Add new graph metric: participation coefficient"
   git tag v0.1.1
   git push origin main --tags
   ```

2. **Update pipelines to use new version:**
   ```bash
   cd mrtrix3_demon_addon
   pip install -e ../connectivity_shared  # For local dev
   # OR update requirements.txt with new version
   ```

3. **Test the integration:**
   ```bash
   bash validation/run_all_validations.sh
   ```

### When Pipeline Changes Require Shared Updates

1. Make changes in `connectivity_shared` first
2. Test with validation suite
3. Update pipelines to use new functionality
4. Commit both changes with related commit messages

## Git Ignore Recommendations

Create `.gitignore` in each repository:

```gitignore
# Python
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/
.eggs/

# Virtual environments
.venv/
venv/
env/

# IDE
.idea/
.vscode/
*.swp

# OS
.DS_Store
Thumbs.db

# Pipeline-specific
*.nii
*.nii.gz
*.mgz
*.tck
*.mif
logs/
jobs/
data/

# Jupyter
.ipynb_checkpoints/

# Testing
.pytest_cache/
.coverage
htmlcov/
```

## Release Checklist

Before releasing a new version:

1. **Run all tests:**
   ```bash
   bash validation/run_all_validations.sh
   ```

2. **Update version numbers:**
   - `connectivity_shared/pyproject.toml`
   - Pipeline version constants

3. **Update documentation:**
   - README.md
   - CHANGELOG.md (if maintained)

4. **Create release tag:**
   ```bash
   git tag -a v0.1.0 -m "Release v0.1.0: Initial harmonized release"
   git push origin v0.1.0
   ```

5. **Update dependent repositories:**
   - Update version requirements
   - Test with new shared package version

## Handling Conflicts

### Common Conflict Scenarios

1. **Both pipelines modify shared code:**
   - Make changes in `connectivity_shared` first
   - Test with both pipelines before merging

2. **Incompatible changes:**
   - Use feature flags or version checks
   - Example:
     ```python
     from connectivity_shared import __version__
     if __version__ >= "0.2.0":
         use_new_feature()
     else:
         use_legacy_behavior()
     ```

3. **Breaking changes:**
   - Increment MAJOR version
   - Document migration path
   - Consider deprecation warnings

## CI/CD Considerations

### GitHub Actions Example

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          pip install -e ./connectivity_shared
          pip install pytest

      - name: Run tests
        run: |
          bash validation/run_all_validations.sh
```

## Best Practices

1. **Commit messages:** Use conventional commits
   ```
   feat: add Destrieux atlas support to fMRI pipeline
   fix: correct matrix loading for headerless CSV
   docs: update usage guide with new examples
   test: add validation for atlas consistency
   ```

2. **Pull requests:** Always include:
   - Description of changes
   - Test results
   - Breaking change warnings

3. **Code review:** Required for changes to:
   - `connectivity_shared` (affects both pipelines)
   - Core processing logic
   - Output formats

4. **Documentation:** Update docs when:
   - Adding new features
   - Changing APIs
   - Fixing user-facing bugs
