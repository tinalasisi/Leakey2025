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

set -e  # Exit on error

echo "============================================================"
echo "DensePose Demo Job"
echo "Started: $(date)"
echo "Node: $(hostname)"
echo "============================================================"

# Load modules
module load gcc/13.2.0

# Conda should already be active from ~/.bashrc

# Check GPU
echo ""
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total --format=csv

# Check Python/torch
echo ""
echo "Python: $(python --version)"
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# Setup paths
PROJECT_DIR=/nfs/turbo/lsa-tlasisi1/tlasisi/Leakey2025
DETECTRON2_DIR=/nfs/turbo/lsa-tlasisi1/tlasisi/detectron2_repo
cd $PROJECT_DIR

# Clone detectron2 if needed (for configs and apply_net.py)
if [ ! -d "$DETECTRON2_DIR" ]; then
    echo ""
    echo "Cloning detectron2 repo for configs..."
    cd /nfs/turbo/lsa-tlasisi1/tlasisi
    git clone --depth 1 https://github.com/facebookresearch/detectron2.git detectron2_repo
fi

# Run DensePose using official apply_net.py
echo ""
echo "============================================================"
echo "Running DensePose-Chimps on test image..."
echo "============================================================"

cd $DETECTRON2_DIR/projects/DensePose

# Dump raw predictions to pickle file
python apply_net.py dump \
    configs/cse/densepose_rcnn_R_50_FPN_soft_chimps_finetune_4k.yaml \
    https://dl.fbaipublicfiles.com/densepose/cse/densepose_rcnn_R_50_FPN_soft_chimps_finetune_4k/253146869/model_final_52f649.pkl \
    $PROJECT_DIR/data/chimp_test.jpg \
    --output $PROJECT_DIR/output/densepose_results.pkl \
    -v

# Also create visualization
python apply_net.py show \
    configs/cse/densepose_rcnn_R_50_FPN_soft_chimps_finetune_4k.yaml \
    https://dl.fbaipublicfiles.com/densepose/cse/densepose_rcnn_R_50_FPN_soft_chimps_finetune_4k/253146869/model_final_52f649.pkl \
    $PROJECT_DIR/data/chimp_test.jpg \
    dp_contour,bbox \
    --output $PROJECT_DIR/output/chimp_densepose_viz.png \
    -v

echo ""
echo "============================================================"
echo "Results saved to:"
echo "  - output/densepose_results.pkl (raw predictions)"
echo "  - output/chimp_densepose_viz.png (visualization)"
echo ""
echo "Job finished: $(date)"
echo "============================================================"