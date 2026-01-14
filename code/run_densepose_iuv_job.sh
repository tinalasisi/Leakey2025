#!/bin/bash
#SBATCH --job-name=densepose_iuv
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=output/densepose_iuv_%j.log
#SBATCH --error=output/densepose_iuv_%j.err

# Run IUV DensePose model (human model, but may detect chimps)
# This outputs I, U, V coordinates directly (not embeddings)

cd /nfs/turbo/lsa-tlasisi1/tlasisi/detectron2_repo/projects/DensePose

python apply_net.py dump \
    configs/densepose_rcnn_R_50_FPN_s1x.yaml \
    https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl \
    /nfs/turbo/lsa-tlasisi1/tlasisi/Leakey2025/data/chimp_test.jpg \
    --output /nfs/turbo/lsa-tlasisi1/tlasisi/Leakey2025/output/densepose_iuv_results.pkl \
    -v

echo "Done: $(date)"
