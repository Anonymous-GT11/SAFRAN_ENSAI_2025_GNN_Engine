#!/bin/bash

# Setup script for Smart Data Research Project
# This script creates a virtual environment with Python 3.12.4 and installs all required packages
#
# IMPORTANT: This installs PyTorch with BUNDLED CUDA (no system CUDA needed!)
# You ONLY need the NVIDIA driver installed: sudo ubuntu-drivers install

set -e  # Exit on any error

echo "========================================"
echo "Smart Data Research Project Setup"
echo "========================================"
echo ""
echo "This will install PyTorch with bundled CUDA 11.8"
echo "You do NOT need to install system CUDA toolkit!"
echo ""

# Check for NVIDIA driver
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA driver detected:"
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
    echo ""
else
    echo "⚠️  WARNING: nvidia-smi not found!"
    echo "   You need to install NVIDIA driver first:"
    echo "   sudo ubuntu-drivers install"
    echo "   sudo reboot"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "========================================"
echo "Setting up Python environment..."
echo "========================================"

# Check if pyenv is installed
if ! command -v pyenv &> /dev/null; then
    echo "Error: pyenv is not installed. Please install pyenv first."
    echo "Visit: https://github.com/pyenv/pyenv#installation"
    exit 1
fi

# Set Python version to 3.12.4
PYTHON_VERSION="3.12.4"
echo "Setting Python version to $PYTHON_VERSION..."

# Check if the Python version is installed
if ! pyenv versions | grep -q "$PYTHON_VERSION"; then
    echo "Python $PYTHON_VERSION is not installed. Installing..."
    pyenv install $PYTHON_VERSION
else
    echo "Python $PYTHON_VERSION is already installed."
fi

# Set local Python version
pyenv local $PYTHON_VERSION

# Remove existing venv if it exists
if [ -d "venv" ]; then
    echo "Removing existing virtual environment..."
    rm -rf venv
fi

# Create virtual environment
echo "Creating virtual environment..."
python -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install packages from requirements.txt
if [ -f "requirements.txt" ]; then
    echo "Installing packages from requirements.txt..."
    pip install -r requirements.txt
else
    echo "Warning: requirements.txt not found. Installing core packages..."
    # Install PyTorch with CUDA support
    echo "Installing PyTorch with CUDA 11.8 support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    
    # Install PyTorch Geometric
    echo "Installing PyTorch Geometric..."
    pip install torch-geometric
    
    # Install PyTorch Geometric Temporal
    echo "Installing PyTorch Geometric Temporal..."
    pip install torch-geometric-temporal
    
    # Generate requirements.txt
    pip freeze > requirements.txt
    echo "Generated requirements.txt with installed packages."
fi

echo ""
echo "================================"
echo "Setup complete!"
echo "================================"
echo ""
echo "To activate the virtual environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "Python version:"
python --version
echo ""
echo "Installed packages:"
pip list
