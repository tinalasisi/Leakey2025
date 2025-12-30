#!/usr/bin/env python3
"""
Create body part collage from DensePose predictions.

Simple visualization showing:
- Anterior (front) body parts
- Posterior (back) body parts
- Coverage statistics

For grant rebuttal demo - shows body surface extraction, not UV mapping.
"""

import sys
sys.path.insert(0, '/nfs/turbo/lsa-tlasisi1/tlasisi/detectron2_repo/projects/DensePose')

import torch
import pickle
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

# DensePose 24 body parts grouped by anterior/posterior
BODY_PARTS = {
    # Posterior (back) parts
    'posterior': {
        1: 'Torso',
        7: 'R-UpperLeg', 8: 'L-UpperLeg',
        11: 'R-LowerLeg', 12: 'L-LowerLeg', 
        15: 'L-UpperArm', 16: 'R-UpperArm',
        19: 'L-LowerArm', 20: 'R-LowerArm',
        23: 'R-Head', 24: 'L-Head',
    },
    # Anterior (front) parts
    'anterior': {
        2: 'Torso',
        3: 'R-Hand', 4: 'L-Hand',
        5: 'L-Foot', 6: 'R-Foot',
        9: 'R-UpperLeg', 10: 'L-UpperLeg',
        13: 'R-LowerLeg', 14: 'L-LowerLeg',
        17: 'L-UpperArm', 18: 'R-UpperArm',
        21: 'L-LowerArm', 22: 'R-LowerArm',
    }
}

# Template layout for body parts (row, col) in a grid
# Organized anatomically: head top, arms sides, torso center, legs bottom
TEMPLATE_LAYOUT = {
    'anterior': {
        'Head': (0, 1),
        'R-UpperArm': (1, 0), 'L-UpperArm': (1, 2),
        'R-LowerArm': (2, 0), 'L-LowerArm': (2, 2),
        'Torso': (1, 1),
        'R-Hand': (3, 0), 'L-Hand': (3, 2),
        'R-UpperLeg': (2, 1), 'L-UpperLeg': (2, 1),  # Share cell
        'R-LowerLeg': (3, 1), 'L-LowerLeg': (3, 1),  # Share cell
        'R-Foot': (4, 0), 'L-Foot': (4, 2),
    },
    'posterior': {
        'R-Head': (0, 1), 'L-Head': (0, 1),  # Share cell
        'R-UpperArm': (1, 0), 'L-UpperArm': (1, 2),
        'R-LowerArm': (2, 0), 'L-LowerArm': (2, 2),
        'Torso': (1, 1),
        'R-UpperLeg': (2, 1), 'L-UpperLeg': (2, 1),
        'R-LowerLeg': (3, 1), 'L-LowerLeg': (3, 1),
    }
}


def load_results(pkl_path):
    """Load DensePose results."""
    try:
        data = torch.load(pkl_path, map_location='cpu', weights_only=False)
    except TypeError:
        data = torch.load(pkl_path, map_location='cpu')
    return data


def extract_body_parts(image_rgb, I_map, box):
    """
    Extract pixels for each body part.
    
    Returns dict mapping part_id -> cropped pixels (as image with alpha)
    """
    x1, y1, x2, y2 = box
    h, w = I_map.shape
    
    parts = {}
    
    for part_id in range(1, 25):
        mask = (I_map == part_id)
        if not mask.any():
            continue
        
        # Find bounding box of this part
        ys, xs = np.where(mask)
        py1, py2 = ys.min(), ys.max() + 1
        px1, px2 = xs.min(), xs.max() + 1
        
        # Extract the part with its mask
        part_mask = mask[py1:py2, px1:px2]
        
        # Map to image coordinates
        img_y1 = y1 + int(py1 * (y2 - y1) / h)
        img_y2 = y1 + int(py2 * (y2 - y1) / h)
        img_x1 = x1 + int(px1 * (x2 - x1) / w)
        img_x2 = x1 + int(px2 * (x2 - x1) / w)
        
        # Clip to image bounds
        img_h, img_w = image_rgb.shape[:2]
        img_y1 = max(0, min(img_y1, img_h))
        img_y2 = max(0, min(img_y2, img_h))
        img_x1 = max(0, min(img_x1, img_w))
        img_x2 = max(0, min(img_x2, img_w))
        
        if img_y2 <= img_y1 or img_x2 <= img_x1:
            continue
        
        # Extract pixels
        part_img = image_rgb[img_y1:img_y2, img_x1:img_x2].copy()
        
        # Resize mask to match
        part_mask_resized = cv2.resize(
            part_mask.astype(np.uint8), 
            (part_img.shape[1], part_img.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )
        
        # Create RGBA image
        part_rgba = np.zeros((part_img.shape[0], part_img.shape[1], 4), dtype=np.uint8)
        part_rgba[:, :, :3] = part_img
        part_rgba[:, :, 3] = part_mask_resized * 255
        
        parts[part_id] = {
            'image': part_rgba,
            'pixel_count': part_mask_resized.sum(),
        }
    
    return parts


def create_body_collage(parts, view='anterior', cell_size=100):
    """
    Create a body part collage for one view (anterior or posterior).
    
    Returns:
        collage: RGB image
        coverage: dict of region -> percentage filled
    """
    grid_rows, grid_cols = 5, 3
    collage = np.zeros((grid_rows * cell_size, grid_cols * cell_size, 3), dtype=np.uint8)
    collage[:] = 40  # Dark gray background
    
    coverage = {}
    part_mapping = BODY_PARTS[view]
    
    for part_id, region_name in part_mapping.items():
        if part_id not in parts:
            coverage[region_name] = 0
            continue
        
        part_data = parts[part_id]
        part_img = part_data['image']
        
        # Get grid position for this region
        layout = TEMPLATE_LAYOUT[view]
        if region_name not in layout:
            continue
        row, col = layout[region_name]
        
        # Resize part to fit cell while maintaining aspect ratio
        ph, pw = part_img.shape[:2]
        scale = min(cell_size / pw, cell_size / ph) * 0.9
        new_w = int(pw * scale)
        new_h = int(ph * scale)
        
        if new_w > 0 and new_h > 0:
            part_resized = cv2.resize(part_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Center in cell
            y_offset = row * cell_size + (cell_size - new_h) // 2
            x_offset = col * cell_size + (cell_size - new_w) // 2
            
            # Blend using alpha
            alpha = part_resized[:, :, 3:4] / 255.0
            rgb = part_resized[:, :, :3]
            
            roi = collage[y_offset:y_offset+new_h, x_offset:x_offset+new_w]
            collage[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = (
                alpha * rgb + (1 - alpha) * roi
            ).astype(np.uint8)
        
        coverage[region_name] = 100  # Mark as present
    
    return collage, coverage


def create_figure(image_path, results, output_path):
    """Create the visualization figure."""
    
    # Load image
    image = cv2.imread(str(image_path))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    
    # Get predictions
    result = results[0]
    boxes = result['pred_boxes_XYXY']
    if hasattr(boxes, 'cpu'):
        boxes = boxes.cpu().numpy()
    box = boxes[0].astype(int)
    
    dp = result['pred_densepose'][0]
    labels = dp.labels.cpu().numpy() if hasattr(dp.labels, 'cpu') else dp.labels
    
    # Extract body parts
    parts = extract_body_parts(image_rgb, labels, box)
    
    print(f"Extracted {len(parts)} body parts:")
    for part_id in sorted(parts.keys()):
        # Find which view this belongs to
        view = 'anterior' if part_id in BODY_PARTS['anterior'] else 'posterior'
        name = BODY_PARTS[view].get(part_id, f'Part {part_id}')
        print(f"  {part_id}: {name} ({view}) - {parts[part_id]['pixel_count']} pixels")
    
    # Create collages
    anterior_collage, anterior_cov = create_body_collage(parts, 'anterior', cell_size=120)
    posterior_collage, posterior_cov = create_body_collage(parts, 'posterior', cell_size=120)
    
    # Count coverage
    anterior_parts = set(BODY_PARTS['anterior'].values())
    posterior_parts = set(BODY_PARTS['posterior'].values())
    
    anterior_detected = sum(1 for r in anterior_cov if anterior_cov[r] > 0)
    posterior_detected = sum(1 for r in posterior_cov if posterior_cov[r] > 0)
    
    anterior_pct = 100 * anterior_detected / len(anterior_parts) if anterior_parts else 0
    posterior_pct = 100 * posterior_detected / len(posterior_parts) if posterior_parts else 0
    
    # Create segmentation overlay
    x1, y1, x2, y2 = box
    box_h, box_w = y2 - y1, x2 - x1
    labels_resized = cv2.resize(labels.astype(np.float32), (box_w, box_h), 
                                 interpolation=cv2.INTER_NEAREST).astype(np.uint8)
    
    I_full = np.zeros((h, w), dtype=np.uint8)
    y1c, y2c = max(0, y1), min(h, y2)
    x1c, x2c = max(0, x1), min(w, x2)
    sy1, sy2 = y1c - y1, box_h - (y2 - y2c)
    sx1, sx2 = x1c - x1, box_w - (x2 - x2c)
    I_full[y1c:y2c, x1c:x2c] = labels_resized[sy1:sy2, sx1:sx2]
    
    # Create figure
    fig = plt.figure(figsize=(16, 10))
    
    # Layout: 2 rows
    # Top: Original | Segmentation
    # Bottom: Anterior template | Posterior template
    
    ax1 = fig.add_subplot(2, 3, 1)
    ax2 = fig.add_subplot(2, 3, 2)
    ax3 = fig.add_subplot(2, 3, 3)
    ax4 = fig.add_subplot(2, 3, 4)
    ax5 = fig.add_subplot(2, 3, 5)
    ax6 = fig.add_subplot(2, 3, 6)
    
    # 1. Original image
    ax1.imshow(image_rgb)
    rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='lime', linewidth=2)
    ax1.add_patch(rect)
    ax1.set_title('A. Input Image', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # 2. Body part segmentation with colors
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    colors2 = plt.cm.tab20b(np.linspace(0, 1, 5))
    all_colors = np.vstack([np.array([[0,0,0,1]]), colors, colors2[:4]])  # 25 colors total
    cmap = ListedColormap(all_colors)
    
    ax2.imshow(I_full, cmap=cmap, vmin=0, vmax=24)
    ax2.set_title('B. Body Part Segmentation', fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    # 3. Segmentation overlay on image
    overlay = image_rgb.copy()
    for part_id in range(1, 25):
        mask = I_full == part_id
        if mask.any():
            color = (np.array(all_colors[part_id][:3]) * 255).astype(np.uint8)
            overlay[mask] = overlay[mask] * 0.5 + color * 0.5
    ax3.imshow(overlay.astype(np.uint8))
    ax3.set_title('C. Segmentation Overlay', fontsize=12, fontweight='bold')
    ax3.axis('off')
    
    # 4. Anterior collage
    ax4.imshow(anterior_collage)
    ax4.set_title(f'D. Anterior (Front) View\n{anterior_detected}/{len(anterior_parts)} regions', 
                  fontsize=12, fontweight='bold')
    ax4.axis('off')
    
    # 5. Posterior collage
    ax5.imshow(posterior_collage)
    ax5.set_title(f'E. Posterior (Back) View\n{posterior_detected}/{len(posterior_parts)} regions',
                  fontsize=12, fontweight='bold')
    ax5.axis('off')
    
    # 6. Coverage summary
    ax6.axis('off')
    
    # Create coverage bar chart
    regions = ['Head', 'Torso', 'UpperArm', 'LowerArm', 'Hand', 'UpperLeg', 'LowerLeg', 'Foot']
    anterior_vals = []
    posterior_vals = []
    
    for region in regions:
        # Check if any version of this region was detected
        ant_found = any(region in r and anterior_cov.get(r, 0) > 0 
                       for r in ['L-'+region, 'R-'+region, region])
        post_found = any(region in r and posterior_cov.get(r, 0) > 0 
                        for r in ['L-'+region, 'R-'+region, region])
        anterior_vals.append(1 if ant_found else 0)
        posterior_vals.append(1 if post_found else 0)
    
    x = np.arange(len(regions))
    width = 0.35
    
    ax6_inner = fig.add_axes([0.68, 0.12, 0.28, 0.35])
    ax6_inner.barh(x - width/2, anterior_vals, width, label='Anterior', color='steelblue')
    ax6_inner.barh(x + width/2, posterior_vals, width, label='Posterior', color='coral')
    ax6_inner.set_yticks(x)
    ax6_inner.set_yticklabels(regions)
    ax6_inner.set_xlim(0, 1.2)
    ax6_inner.set_xlabel('Detected')
    ax6_inner.set_title('F. Coverage by Region', fontweight='bold')
    ax6_inner.legend(loc='lower right', fontsize=8)
    ax6_inner.set_xticks([0, 1])
    ax6_inner.set_xticklabels(['No', 'Yes'])
    
    # Title
    fig.suptitle('DensePose Body Surface Extraction Demo', fontsize=14, fontweight='bold', y=0.98)
    
    # Caption
    total_parts = len([p for p in parts])
    caption = f"Detected {total_parts} body part regions from single image. This demonstrates coordinate-based surface mapping, not image warping."
    fig.text(0.5, 0.02, caption, ha='center', fontsize=10, style='italic')
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\nSaved figure to: {output_path}")


def main():
    project_dir = Path("/nfs/turbo/lsa-tlasisi1/tlasisi/Leakey2025")
    
    pkl_path = project_dir / "output" / "densepose_iuv_results.pkl"
    image_path = project_dir / "data" / "chimp_test.jpg"
    output_path = project_dir / "output" / "densepose_body_collage.png"
    
    if not pkl_path.exists():
        print(f"Results file not found: {pkl_path}")
        return
    
    print("="*60)
    print("Creating Body Part Collage")
    print("="*60)
    
    results = load_results(pkl_path)
    create_figure(image_path, results, output_path)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
