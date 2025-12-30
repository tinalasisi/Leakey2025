#!/usr/bin/env python3
"""
Process DensePose pickle output and build texture atlas.

Run AFTER the SLURM job completes:
    python code/process_densepose_results.py

This script:
1. Loads the pickle file from apply_net.py dump
2. Extracts the IUV/CSE predictions
3. Builds a texture atlas
4. Creates the 4-panel figure for grant rebuttal
"""

import pickle
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import sys

# Add detectron2 DensePose to path for unpickling
sys.path.insert(0, '/nfs/turbo/lsa-tlasisi1/tlasisi/detectron2_repo/projects/DensePose')

import torch

# Import DensePose structures (needed for unpickling)
try:
    from densepose.structures import DensePoseEmbeddingPredictorOutput
except ImportError:
    print("Warning: Could not import DensePose structures")


def load_densepose_results(pkl_path):
    """Load results from apply_net.py dump output."""
    # Try torch.load first (handles PyTorch tensors)
    try:
        # weights_only=False needed for PyTorch 2.0+
        data = torch.load(pkl_path, map_location='cpu', weights_only=False)
    except TypeError:
        # Older PyTorch versions don't have weights_only
        data = torch.load(pkl_path, map_location='cpu')
    except Exception as e:
        print(f"torch.load failed: {e}")
        # Fall back to pickle
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
    
    print(f"Loaded {len(data)} image results")
    
    # Each item in data is a dict with keys like:
    # - file_name
    # - scores
    # - pred_boxes_XYXY
    # - pred_densepose (list of DensePoseChartPredictorOutput or CSE results)
    
    for i, result in enumerate(data):
        print(f"\nImage {i}: {result.get('file_name', 'unknown')}")
        print(f"  Keys: {result.keys()}")
        
        if 'pred_boxes_XYXY' in result:
            boxes = result['pred_boxes_XYXY']
            print(f"  Detected {len(boxes)} instances")
            
        if 'pred_densepose' in result:
            dp = result['pred_densepose']
            print(f"  DensePose predictions: {len(dp)} instances")
            for j, pred in enumerate(dp):
                print(f"    Instance {j}: {type(pred)}")
                if hasattr(pred, 'labels'):
                    print(f"      labels shape: {pred.labels.shape}")
                if hasattr(pred, 'uv'):
                    print(f"      uv shape: {pred.uv.shape}")
                # CSE models have embeddings instead of IUV
                if hasattr(pred, 'embedding'):
                    print(f"      embedding shape: {pred.embedding.shape}")
                if hasattr(pred, 'coarse_segm'):
                    print(f"      coarse_segm shape: {pred.coarse_segm.shape}")
    
    return data


def extract_iuv_from_cse(result, image_shape):
    """
    Extract IUV-like data from CSE predictions.
    
    CSE models predict continuous embeddings rather than discrete IUV.
    We can still visualize the coarse segmentation.
    """
    h, w = image_shape[:2]
    
    if len(result.get('pred_densepose', [])) == 0:
        print("No DensePose predictions found!")
        return None, None, None, None
    
    # Get first instance
    boxes = result['pred_boxes_XYXY']
    dp = result['pred_densepose'][0]
    
    # Handle PyTorch tensors
    if hasattr(boxes, 'cpu'):
        boxes = boxes.cpu().numpy()
    box = boxes[0].astype(int)
    x1, y1, x2, y2 = box
    
    print(f"Bounding box: [{x1}, {y1}, {x2}, {y2}]")
    
    # Check what type of prediction we have
    if hasattr(dp, 'labels'):
        # IUV-style predictions
        labels = dp.labels.cpu().numpy() if hasattr(dp.labels, 'cpu') else dp.labels
        uv = dp.uv.cpu().numpy() if hasattr(dp.uv, 'cpu') else dp.uv
        
        print(f"IUV predictions found")
        print(f"  Labels shape: {labels.shape}, unique: {np.unique(labels)}")
        print(f"  UV shape: {uv.shape}")
        
        return labels, uv[0], uv[1], box
        
    elif hasattr(dp, 'coarse_segm'):
        # CSE predictions - use coarse segmentation
        coarse = dp.coarse_segm
        if hasattr(coarse, 'cpu'):
            coarse = coarse.cpu().numpy()
        
        print(f"CSE predictions found (coarse segmentation)")
        print(f"  Coarse segm shape: {coarse.shape}")
        
        # Coarse segm is (1, num_classes, H, W) - squeeze and take argmax
        if len(coarse.shape) == 4:
            coarse = coarse[0]  # Remove batch dim -> (num_classes, H, W)
        if len(coarse.shape) == 3:
            labels = np.argmax(coarse, axis=0)  # -> (H, W)
        else:
            labels = coarse
            
        print(f"  Labels shape: {labels.shape}, unique: {np.unique(labels)}")
        
        # CSE also has embeddings - we could visualize these
        if hasattr(dp, 'embedding'):
            emb = dp.embedding
            if hasattr(emb, 'cpu'):
                emb = emb.cpu().numpy()
            print(f"  Embedding shape: {emb.shape}")
        
        # CSE doesn't have direct UV - we can't build a proper atlas
        return labels, None, None, box
    
    else:
        print(f"Unknown prediction type. Attributes: {dir(dp)}")
        return None, None, None, None


def build_texture_atlas_iuv(image, I_map, U_map, V_map, atlas_size=512):
    """
    Build UV texture atlas from IUV predictions.
    
    Only works with IUV-style predictions, not CSE.
    """
    if U_map is None or V_map is None:
        print("Cannot build texture atlas without UV coordinates (CSE model)")
        return None, None
    
    # Convert BGR to RGB
    if len(image.shape) == 3 and image.shape[2] == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image
    
    grid_size = 4
    cell_size = atlas_size // grid_size
    
    atlas = np.zeros((atlas_size, atlas_size, 3), dtype=np.uint8)
    coverage = np.zeros((atlas_size, atlas_size), dtype=np.float32)
    
    h, w = I_map.shape
    
    for y in range(h):
        for x in range(w):
            part_id = int(I_map[y, x])
            if part_id == 0:
                continue
            
            u = np.clip(float(U_map[y, x]), 0, 1)
            v = np.clip(float(V_map[y, x]), 0, 1)
            
            cell_row = (part_id - 1) // grid_size
            cell_col = (part_id - 1) % grid_size
            
            atlas_x = int(cell_col * cell_size + u * (cell_size - 1))
            atlas_y = int(cell_row * cell_size + v * (cell_size - 1))
            
            if 0 <= atlas_x < atlas_size and 0 <= atlas_y < atlas_size:
                # Map pixel from original image
                # We need to map from the bbox-relative coordinates to image coordinates
                atlas[atlas_y, atlas_x] = image_rgb[y, x]
                coverage[atlas_y, atlas_x] = 1.0
    
    return atlas, coverage


def create_4panel_figure(image_path, results, output_path):
    """
    Create the 4-panel rebuttal figure.
    
    A. Original image
    B. Body part segmentation (or foreground mask for CSE)
    C. Surface atlas (if IUV available) or embedding visualization
    D. Coverage map (if IUV available) or detection overlay
    """
    # Load original image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Could not load image: {image_path}")
        return
    
    h, w = image.shape[:2]
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Extract predictions
    result = results[0]  # First image
    labels, U, V, box = extract_iuv_from_cse(result, image.shape)
    
    if labels is None:
        print("No predictions to visualize")
        return
    
    x1, y1, x2, y2 = box
    box_h, box_w = y2 - y1, x2 - x1
    
    # Resize labels to bounding box and place in full image
    labels_resized = cv2.resize(labels.astype(np.float32), (box_w, box_h), 
                                 interpolation=cv2.INTER_NEAREST).astype(np.uint8)
    
    I_full = np.zeros((h, w), dtype=np.uint8)
    # Clip to image bounds
    y1c, y2c = max(0, y1), min(h, y2)
    x1c, x2c = max(0, x1), min(w, x2)
    sy1, sy2 = y1c - y1, box_h - (y2 - y2c)
    sx1, sx2 = x1c - x1, box_w - (x2 - x2c)
    I_full[y1c:y2c, x1c:x2c] = labels_resized[sy1:sy2, sx1:sx2]
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # A: Original image with bounding box
    axes[0, 0].imshow(image_rgb)
    rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='lime', linewidth=2)
    axes[0, 0].add_patch(rect)
    axes[0, 0].set_title('A. Original Image + Detection', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # B: Segmentation mask overlay
    # Create colored overlay
    overlay = image_rgb.copy()
    mask = I_full > 0
    overlay[mask] = overlay[mask] * 0.5 + np.array([0, 255, 0]) * 0.5  # Green tint
    
    axes[0, 1].imshow(overlay.astype(np.uint8))
    axes[0, 1].set_title('B. Detected Chimp (Segmentation Mask)', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    # C: Segmentation as binary mask
    axes[1, 0].imshow(I_full, cmap='Greens')
    axes[1, 0].set_title('C. Binary Segmentation Mask', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    # D: Cropped detection
    crop = image_rgb[y1c:y2c, x1c:x2c]
    axes[1, 1].imshow(crop)
    axes[1, 1].set_title('D. Detected Region (Cropped)', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    
    # Add info text
    fig.suptitle('DensePose-Chimps Detection Result\n(CSE Model - Continuous Surface Embeddings)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Add caption
    scores = result['scores']
    if hasattr(scores, 'cpu'):
        scores = scores.cpu().numpy()
    caption = (f"Detection confidence: {scores[0]:.2f}\n"
               f"Bounding box: [{x1}, {y1}, {x2}, {y2}]\n"
               f"Model: densepose_rcnn_R_50_FPN_soft_chimps_finetune_4k")
    fig.text(0.5, 0.02, caption, ha='center', fontsize=10, style='italic')
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved 4-panel figure to: {output_path}")


def main():
    project_dir = Path("/nfs/turbo/lsa-tlasisi1/tlasisi/Leakey2025")
    
    pkl_path = project_dir / "output" / "densepose_results.pkl"
    image_path = project_dir / "data" / "chimp_test.jpg"
    output_path = project_dir / "output" / "densepose_4panel_figure.png"
    
    if not pkl_path.exists():
        print(f"Results file not found: {pkl_path}")
        print("Run the SLURM job first: sbatch code/run_densepose_job.sh")
        return
    
    print("="*60)
    print("Processing DensePose Results")
    print("="*60)
    
    # Load results
    results = load_densepose_results(pkl_path)
    
    # Create figure
    create_4panel_figure(image_path, results, output_path)
    
    print("\nDone!")


if __name__ == "__main__":
    main()