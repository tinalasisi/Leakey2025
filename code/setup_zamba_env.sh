#!/bin/bash
# ============================================================
# Setup Zamba DensePose Environment on Great Lakes
# ============================================================
#
# Run this script ONCE to create the environment:
#     bash code/setup_zamba_env.sh
#
# Then activate it before running scripts:
#     source ~/zamba_env/bin/activate
#
# ============================================================

set -e  # Exit on error

echo "============================================================"
echo "Setting up Zamba DensePose Environment"
echo "============================================================"

# Step 1: Load required modules
echo ""
echo "Step 1: Loading modules..."
module load gcc/13.2.0
module load python/3.11

# Check if python 3.11 loaded, if not try alternatives
PYTHON_VERSION=$(python3 --version 2>&1)
if [[ ! "$PYTHON_VERSION" =~ "3.11" ]] && [[ ! "$PYTHON_VERSION" =~ "3.12" ]]; then
    echo "  Trying python/3.12..."
    module load python/3.12 2>/dev/null || true
fi

echo "  ✓ GCC version: $(gcc --version | head -1)"
echo "  ✓ Python version: $(python3 --version)"

# Verify Python version is 3.11+
PYTHON_MAJOR=$(python3 -c "import sys; print(sys.version_info.major)")
PYTHON_MINOR=$(python3 -c "import sys; print(sys.version_info.minor)")
if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 11 ]); then
    echo ""
    echo "ERROR: Zamba requires Python 3.11 or higher"
    echo "  Current version: $(python3 --version)"
    echo ""
    echo "Check available Python modules with:"
    echo "  module avail python"
    echo ""
    echo "Then load the appropriate one, e.g.:"
    echo "  module load python/3.11"
    echo "  module load python/3.12"
    exit 1
fi

# Step 2: Create virtual environment
ENV_DIR="$HOME/zamba_env"

if [ -d "$ENV_DIR" ]; then
    echo ""
    echo "Environment already exists at $ENV_DIR"
    echo "To recreate, first run: rm -rf $ENV_DIR"
    echo ""
    read -p "Activate existing environment and continue? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo ""
    echo "Step 2: Creating virtual environment at $ENV_DIR..."
    python3 -m venv "$ENV_DIR"
    echo "  ✓ Virtual environment created"
fi

# Step 3: Activate and upgrade pip
echo ""
echo "Step 3: Activating environment..."
source "$ENV_DIR/bin/activate"

echo "  Upgrading pip..."
pip install --upgrade pip wheel setuptools

# Step 4: Install Zamba with DensePose
echo ""
echo "Step 4: Installing Zamba with DensePose support..."
echo "  (This may take several minutes)"

pip install "https://github.com/drivendataorg/zamba/releases/latest/download/zamba.tar.gz#egg=zamba[densepose]"

# Step 5: Install additional dependencies
echo ""
echo "Step 5: Installing additional dependencies..."
pip install pandas matplotlib pillow

# Step 6: Verify installation
echo ""
echo "Step 6: Verifying installation..."

if command -v zamba &> /dev/null; then
    echo "  ✓ Zamba CLI available"
    zamba --help | head -5
else
    echo "  ✗ Zamba CLI not found"
    exit 1
fi

# Check densepose
if zamba densepose --help &> /dev/null; then
    echo "  ✓ Zamba DensePose subcommand available"
else
    echo "  ✗ Zamba DensePose not available"
    echo "    Try reinstalling with: pip install 'zamba[densepose]'"
fi

# Step 7: Create activation helper
ACTIVATE_SCRIPT="$HOME/activate_zamba.sh"
cat > "$ACTIVATE_SCRIPT" << 'EOF'
#!/bin/bash
# Quick activation script for Zamba environment
module load gcc/13.2.0
module load python/3.11 2>/dev/null || module load python/3.12 2>/dev/null
source ~/zamba_env/bin/activate
echo "Zamba environment activated"
echo "  Python: $(which python) ($(python --version))"
echo "  Zamba: $(which zamba)"
EOF
chmod +x "$ACTIVATE_SCRIPT"

echo ""
echo "============================================================"
echo "SETUP COMPLETE"
echo "============================================================"
echo ""
echo "To activate the environment in the future, run:"
echo "    source ~/activate_zamba.sh"
echo ""
echo "Or manually:"
echo "    module load gcc/13.2.0"
echo "    module load python/3.10"
echo "    source ~/zamba_env/bin/activate"
echo ""
echo "Then run your scripts:"
echo "    cd /path/to/your/project"
echo "    python code/explore_zamba_densepose.py"
echo ""