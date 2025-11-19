# SAFRAN ENSAI 2025 GNN Engine

Graph Neural Network project for engine performance modeling and fault detection.

## Requirements

- Ubuntu 22.04 LTS
- NVIDIA GPU with driver installed
- Python 3.12.4

## Installation

### 1. Install NVIDIA Driver

```bash
sudo ubuntu-drivers install
sudo reboot
nvidia-smi  # Verify installation
```

### 2. Run Setup Script

```bash
git clone https://github.com/Anonymous-GT11/SAFRAN_ENSAI_2025_GNN_Engine.git
cd SAFRAN_ENSAI_2025_GNN_Engine
chmod +x setup.sh
./setup.sh
```

The script installs Python 3.12.4, creates a virtual environment, and installs all dependencies including PyTorch with bundled CUDA 11.8.

### 3. Activate and Verify

```bash
source venv/bin/activate
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

## Usage

```bash
source venv/bin/activate
jupyter notebook  # Open notebooks in GNN_tutorial/ or engine_simulator/notebook/
```

## Project Structure

```
├── GNN_tutorial/          # GNN learning notebooks
├── engine_simulator/      # Engine performance modeling
├── data/                  # Datasets
└── requirements.txt       # Dependencies
```

## Troubleshooting

**NVIDIA driver not found:**
```bash
sudo ubuntu-drivers install && sudo reboot
```

**CUDA not available in PyTorch:**
```bash
pip show torch  # Should show: 2.7.1+cu118
source venv/bin/activate  # Make sure venv is active
```

**Virtual environment not activated:**
```bash
source venv/bin/activate
```

## Notes

- PyTorch includes bundled CUDA 11.8 - no separate CUDA Toolkit needed
- Tested on NVIDIA RTX 1000 Ada (6GB VRAM)
- Minimum: 6GB GPU VRAM, 8GB RAM, 10GB disk space
