#!/bin/bash
#SBATCH --job-name=densepose_demo
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=output/densepose_%j.log
#SBATCH --error=output/densepose_%j.err

# ============================================================
# DensePose Demo Job
# Submit from project directory:
#     sbatch code/run_densepose_job.sh
# ============================================================

echo "============================================================"
echo "DensePose Demo Job"
echo "Started: $(date)"
echo "Node: $(hostname)"
echo "============================================================"

# Load modules
module load gcc/13.2.0

# Activate conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base

# Check GPU
echo ""
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total --format=csv

# Check Python/torch
echo ""
echo "Python: $(python --version)"
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# Run DensePose on chimp image
echo ""
echo "============================================================"
echo "Running DensePose-Chimps on test image..."
echo "============================================================"

cd /nfs/turbo/lsa-tlasisi1/tlasisi/Leakey2025

python code/run_densepose_image.py data/chimp_test.jpg \
    --output-dir output/densepose_demo \
    --chimps

echo ""
echo "============================================================"
echo "Job finished: $(date)"
echo "============================================================"
