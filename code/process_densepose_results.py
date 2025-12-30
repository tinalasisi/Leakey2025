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
    Extract IUV data from DensePose predictions.
    
    Handles both IUV (DensePoseChartResult) and CSE (DensePoseEmbeddingPredictorOutput).
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
    print(f"Prediction type: {type(dp).__name__}")
    
    # Check what type of prediction we have
    if hasattr(dp, 'labels') and hasattr(dp, 'uv'):
        # IUV-style predictions (DensePoseChartResult)
        labels = dp.labels.cpu().numpy() if hasattr(dp.labels, 'cpu') else dp.labels
        uv = dp.uv.cpu().numpy() if hasattr(dp.uv, 'cpu') else dp.uv
        
        print(f"IUV predictions found!")
        print(f"  Labels (I) shape: {labels.shape}, unique values: {np.unique(labels)}")
        print(f"  UV shape: {uv.shape}, U range: [{uv[0].min():.3f}, {uv[0].max():.3f}], V range: [{uv[1].min():.3f}, {uv[1].max():.3f}]")
        
        return labels, uv[0], uv[1], box
        
    elif hasattr(dp, 'coarse_segm'):
        # CSE predictions - use coarse segmentation
        coarse = dp.coarse_segm
        if hasattr(coarse, 'cpu'):
            coarse = coarse.cpu().numpy()
        
        print(f"CSE predictions found (coarse segmentation only)")
        print(f"  Coarse segm shape: {coarse.shape}")
        
        # Coarse segm is (1, num_classes, H, W) - squeeze and take argmax
        if len(coarse.shape) == 4:
            coarse = coarse[0]  # Remove batch dim -> (num_classes, H, W)
        if len(coarse.shape) == 3:
            labels = np.argmax(coarse, axis=0)  # -> (H, W)
        else:
            labels = coarse
            
        print(f"  Labels shape: {labels.shape}, unique: {np.unique(labels)}")
        
        # CSE doesn't have direct UV
        return labels, None, None, box
    
    else:
        print(f"Unknown prediction type. Attributes: {dir(dp)}")
        return None, None, None, None


def build_texture_atlas_iuv(image_rgb, I_map, U_map, V_map, atlas_size=512):
    """
    Build UV texture atlas from IUV predictions.
    
    Args:
        image_rgb: RGB image (H, W, 3)
        I_map: Body part indices (H, W), values 0-14
        U_map: U coordinates (H, W), values 0-1
        V_map: V coordinates (H, W), values 0-1
        atlas_size: Size of output atlas
    
    Returns:
        atlas: (atlas_size, atlas_size, 3) texture atlas
        coverage: (atlas_size, atlas_size) binary coverage map
    """
    if U_map is None or V_map is None:
        print("Cannot build texture atlas without UV coordinates")
        return None, None
    
    # 14 body parts arranged in 4x4 grid (2 cells unused)
    grid_size = 4
    cell_size = atlas_size // grid_size
    
    atlas = np.zeros((atlas_size, atlas_size, 3), dtype=np.uint8)
    coverage = np.zeros((atlas_size, atlas_size), dtype=np.float32)
    
    h, w = I_map.shape
    
    for y in range(h):
        for x in range(w):
            part_id = int(I_map[y, x])
            if part_id == 0:  # Background
                continue
            
            u = np.clip(float(U_map[y, x]), 0, 0.999)
            v = np.clip(float(V_map[y, x]), 0, 0.999)
            
            # Map part_id (1-14) to grid cell (0-13)
            cell_row = (part_id - 1) // grid_size
            cell_col = (part_id - 1) % grid_size
            
            # Map UV to pixel within cell
            atlas_x = int(cell_col * cell_size + u * cell_size)
            atlas_y = int(cell_row * cell_size + v * cell_size)
            
            if 0 <= atlas_x < atlas_size and 0 <= atlas_y < atlas_size:
                atlas[atlas_y, atlas_x] = image_rgb[y, x]
                coverage[atlas_y, atlas_x] = 1.0
    
    return atlas, coverage


def create_4panel_figure(image_path, results, output_path):
    """
    Create the 4-panel rebuttal figure.
    
    A. Original image
    B. Body part segmentation (I values)
    C. Surface atlas (UV texture map)
    D. Coverage map
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
    
    # The IUV predictions are already at the bounding box size
    # We need to place them in the full image context
    
    # Resize predictions to match bounding box in image
    pred_h, pred_w = labels.shape
    
    # Create full-image arrays
    I_full = np.zeros((h, w), dtype=np.uint8)
    U_full = np.zeros((h, w), dtype=np.float32)
    V_full = np.zeros((h, w), dtype=np.float32)
    
    # Resize and place
    labels_resized = cv2.resize(labels.astype(np.float32), (box_w, box_h), 
                                 interpolation=cv2.INTER_NEAREST).astype(np.uint8)
    
    if U is not None and V is not None:
        U_resized = cv2.resize(U, (box_w, box_h), interpolation=cv2.INTER_LINEAR)
        V_resized = cv2.resize(V, (box_w, box_h), interpolation=cv2.INTER_LINEAR)
    
    # Clip to image bounds
    y1c, y2c = max(0, y1), min(h, y2)
    x1c, x2c = max(0, x1), min(w, x2)
    sy1, sy2 = y1c - y1, box_h - (y2 - y2c)
    sx1, sx2 = x1c - x1, box_w - (x2 - x2c)
    
    I_full[y1c:y2c, x1c:x2c] = labels_resized[sy1:sy2, sx1:sx2]
    if U is not None and V is not None:
        U_full[y1c:y2c, x1c:x2c] = U_resized[sy1:sy2, sx1:sx2]
        V_full[y1c:y2c, x1c:x2c] = V_resized[sy1:sy2, sx1:sx2]
    
    # Build texture atlas
    atlas = None
    coverage = None
    if U is not None and V is not None:
        atlas, coverage = build_texture_atlas_iuv(image_rgb, I_full, U_full, V_full)
        
        # Calculate coverage statistics
        total_pixels = coverage.size
        covered_pixels = np.sum(coverage > 0)
        coverage_pct = 100 * covered_pixels / total_pixels
        print(f"\nTexture atlas coverage: {coverage_pct:.1f}% of UV space")
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    part_names = ['BG', 'Torso', 'RHand', 'LHand', 'LFoot', 'RFoot',
                  'RUpLeg', 'LUpLeg', 'RLoLeg', 'LLoLeg',
                  'RUpArm', 'LUpArm', 'RLoArm', 'LLoArm', 'Head']
    
    # A: Original image with bounding box
    axes[0, 0].imshow(image_rgb)
    rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='lime', linewidth=2)
    axes[0, 0].add_patch(rect)
    axes[0, 0].set_title('A. Original Image + Detection', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # B: Body part segmentation
    num_parts = min(int(labels.max()) + 1, 15)
    colors = plt.cm.tab20(np.linspace(0, 1, 15))
    colors[0] = [0, 0, 0, 1]  # Background = black
    cmap = ListedColormap(colors)
    
    axes[0, 1].imshow(I_full, cmap=cmap, vmin=0, vmax=14)
    axes[0, 1].set_title('B. Body Part Segmentation (I)', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Legend for detected parts
    unique_parts = np.unique(I_full)
    patches = []
    for p in unique_parts:
        if 0 < p < len(part_names):
            patches.append(mpatches.Patch(color=colors[p], label=f'{p}: {part_names[p]}'))
    if patches:
        axes[0, 1].legend(handles=patches, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
    
    # C: Surface atlas
    if atlas is not None:
        axes[1, 0].imshow(atlas)
        axes[1, 0].set_title('C. Surface Atlas (UV Texture Map)', fontsize=14, fontweight='bold')
        # Grid lines for body parts
        cell_size = atlas.shape[0] // 4
        for i in range(1, 4):
            axes[1, 0].axhline(y=i*cell_size, color='white', linewidth=0.5, alpha=0.7)
            axes[1, 0].axvline(x=i*cell_size, color='white', linewidth=0.5, alpha=0.7)
    else:
        axes[1, 0].text(0.5, 0.5, 'Atlas not available', ha='center', va='center', 
                        fontsize=12, transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('C. Surface Atlas', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    # D: Coverage map
    if coverage is not None:
        im = axes[1, 1].imshow(coverage, cmap='YlOrRd', vmin=0, vmax=1)
        axes[1, 1].set_title(f'D. Coverage Map ({coverage_pct:.1f}% observed)', fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04, label='Observed')
        cell_size = coverage.shape[0] // 4
        for i in range(1, 4):
            axes[1, 1].axhline(y=i*cell_size, color='gray', linewidth=0.5, alpha=0.5)
            axes[1, 1].axvline(x=i*cell_size, color='gray', linewidth=0.5, alpha=0.5)
    else:
        axes[1, 1].text(0.5, 0.5, 'Coverage not available', ha='center', va='center',
                        fontsize=12, transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('D. Coverage Map', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    
    # Add title
    fig.suptitle('DensePose: Image → Body Surface Mapping', fontsize=16, fontweight='bold', y=0.98)
    
    # Add caption
    scores = result['scores']
    if hasattr(scores, 'cpu'):
        scores = scores.cpu().numpy()
    caption = f"Detection confidence: {scores[0]:.2f} | This demonstrates coordinate lookup, not image warping."
    fig.text(0.5, 0.02, caption, ha='center', fontsize=10, style='italic')
    
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved 4-panel figure to: {output_path}")


def main():
    project_dir = Path("/nfs/turbo/lsa-tlasisi1/tlasisi/Leakey2025")
    
    # Use IUV results (human model worked on chimp!)
    pkl_path = project_dir / "output" / "densepose_iuv_results.pkl"
    image_path = project_dir / "data" / "chimp_test.jpg"
    output_path = project_dir / "output" / "densepose_4panel_figure.png"
    
    if not pkl_path.exists():
        print(f"Results file not found: {pkl_path}")
        print("Run the SLURM job first: sbatch code/run_densepose_iuv_job.sh")
        return
    
    print("="*60)
    print("Processing DensePose IUV Results")
    print("="*60)
    
    # Load results
    results = load_densepose_results(pkl_path)
    
    # Create figure
    create_4panel_figure(image_path, results, output_path)
    
    print("\nDone!")


if __name__ == "__main__":
    main()