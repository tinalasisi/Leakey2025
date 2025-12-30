#!/usr/bin/env python3
"""
Create body part collage from DensePose predictions.

Simple visualization showing:
- Original + segmentation with legend
- Anterior (front) body parts extracted
- Posterior (back) body parts extracted

For grant rebuttal demo.
"""

import sys
sys.path.insert(0, '/nfs/turbo/lsa-tlasisi1/tlasisi/detectron2_repo/projects/DensePose')

import torch
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

# DensePose 24 body parts - official naming
PART_NAMES = {
    0: 'Background',
    1: 'Torso (back)', 2: 'Torso (front)',
    3: 'Right Hand', 4: 'Left Hand',
    5: 'Left Foot', 6: 'Right Foot',
    7: 'Right Upper Leg (back)', 8: 'Left Upper Leg (back)',
    9: 'Right Upper Leg (front)', 10: 'Left Upper Leg (front)',
    11: 'Right Lower Leg (back)', 12: 'Left Lower Leg (back)',
    13: 'Right Lower Leg (front)', 14: 'Left Lower Leg (front)',
    15: 'Left Upper Arm (back)', 16: 'Right Upper Arm (back)',
    17: 'Left Upper Arm (front)', 18: 'Right Upper Arm (front)',
    19: 'Left Lower Arm (back)', 20: 'Right Lower Arm (back)',
    21: 'Left Lower Arm (front)', 22: 'Right Lower Arm (front)',
    23: 'Right Face', 24: 'Left Face',  # Face = anterior!
}

# Which parts are anterior (front) vs posterior (back)
# Head/face (23, 24) is ANTERIOR - it's the face!
# Hands (3, 4) and feet (5, 6) don't have front/back - putting in anterior
ANTERIOR_PARTS = {2, 3, 4, 5, 6, 9, 10, 13, 14, 17, 18, 21, 22, 23, 24}
POSTERIOR_PARTS = {1, 7, 8, 11, 12, 15, 16, 19, 20}

# Template positions for anterior view (row, col) in 4x3 grid
# Organized: head top-center, arms on sides, torso center, legs bottom
ANTERIOR_LAYOUT = {
    # Row 0: Head/Face, Hands
    23: (0, 1),  # Right Face - combine in center
    24: (0, 1),  # Left Face
    3: (0, 2),   # Right Hand
    4: (0, 0),   # Left Hand
    # Row 1: Upper arms, Torso
    18: (1, 2),  # Right Upper Arm (front)
    17: (1, 0),  # Left Upper Arm (front)
    2: (1, 1),   # Torso (front)
    # Row 2: Lower arms, Upper legs  
    22: (2, 2),  # Right Lower Arm (front)
    21: (2, 0),  # Left Lower Arm (front)
    9: (2, 1),   # Right Upper Leg (front)
    10: (2, 1),  # Left Upper Leg (front)
    # Row 3: Feet, Lower legs
    6: (3, 2),   # Right Foot
    5: (3, 0),   # Left Foot
    13: (3, 1),  # Right Lower Leg (front)
    14: (3, 1),  # Left Lower Leg (front)
}

# Template positions for posterior view
POSTERIOR_LAYOUT = {
    # Row 0: empty (no head from back in this image)
    # Row 1: Upper arms, Torso
    16: (1, 2),  # Right Upper Arm (back)
    15: (1, 0),  # Left Upper Arm (back)
    1: (1, 1),   # Torso (back)
    # Row 2: Lower arms, Upper legs
    20: (2, 2),  # Right Lower Arm (back)
    19: (2, 0),  # Left Lower Arm (back)
    7: (2, 1),   # Right Upper Leg (back)
    8: (2, 1),   # Left Upper Leg (back)
    # Row 3: Lower legs
    11: (3, 1),  # Right Lower Leg (back)
    12: (3, 1),  # Left Lower Leg (back)
}


def load_results(pkl_path):
    """Load DensePose results."""
    data = torch.load(pkl_path, map_location='cpu', weights_only=False)
    return data


def get_colors():
    """Get consistent colors for all 25 parts."""
    colors = np.zeros((25, 4))
    colors[0] = [0, 0, 0, 1]  # Background black
    
    # Assign colors by body region for visual grouping
    colors[1] = colors[2] = [0.8, 0.4, 0.4, 1]      # Torso - red-ish
    colors[3] = colors[4] = [0.4, 0.8, 0.4, 1]      # Hands - green
    colors[5] = colors[6] = [0.4, 0.4, 0.8, 1]      # Feet - blue
    colors[7] = colors[8] = [0.9, 0.6, 0.3, 1]      # Upper leg back - orange
    colors[9] = colors[10] = [0.9, 0.7, 0.4, 1]     # Upper leg front - light orange
    colors[11] = colors[12] = [0.6, 0.3, 0.6, 1]    # Lower leg back - purple
    colors[13] = colors[14] = [0.7, 0.4, 0.7, 1]    # Lower leg front - light purple
    colors[15] = colors[16] = [0.3, 0.6, 0.6, 1]    # Upper arm back - teal
    colors[17] = colors[18] = [0.4, 0.7, 0.7, 1]    # Upper arm front - light teal
    colors[19] = colors[20] = [0.6, 0.6, 0.3, 1]    # Lower arm back - olive
    colors[21] = colors[22] = [0.7, 0.7, 0.4, 1]    # Lower arm front - light olive
    colors[23] = colors[24] = [0.9, 0.9, 0.5, 1]    # Head - yellow
    
    return colors


def extract_part_pixels(image_rgb, I_map, box, part_id):
    """
    Extract pixels for a specific body part.
    
    Returns RGBA image of just that part, or None if not present.
    """
    mask = (I_map == part_id)
    if not mask.any():
        return None
    
    x1, y1, x2, y2 = box
    pred_h, pred_w = I_map.shape
    img_h, img_w = image_rgb.shape[:2]
    
    # Find bounding box of this part in prediction coords
    ys, xs = np.where(mask)
    py1, py2 = ys.min(), ys.max() + 1
    px1, px2 = xs.min(), xs.max() + 1
    
    # Map to image coordinates
    scale_y = (y2 - y1) / pred_h
    scale_x = (x2 - x1) / pred_w
    
    img_y1 = int(y1 + py1 * scale_y)
    img_y2 = int(y1 + py2 * scale_y)
    img_x1 = int(x1 + px1 * scale_x)
    img_x2 = int(x1 + px2 * scale_x)
    
    # Clip to image bounds
    img_y1 = max(0, min(img_y1, img_h))
    img_y2 = max(0, min(img_y2, img_h))
    img_x1 = max(0, min(img_x1, img_w))
    img_x2 = max(0, min(img_x2, img_w))
    
    if img_y2 <= img_y1 or img_x2 <= img_x1:
        return None
    
    # Extract image region
    part_img = image_rgb[img_y1:img_y2, img_x1:img_x2].copy()
    
    # Get corresponding mask region and resize to match
    part_mask = mask[py1:py2, px1:px2]
    part_mask_resized = cv2.resize(
        part_mask.astype(np.uint8),
        (part_img.shape[1], part_img.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )
    
    # Create RGBA
    rgba = np.zeros((part_img.shape[0], part_img.shape[1], 4), dtype=np.uint8)
    rgba[:, :, :3] = part_img
    rgba[:, :, 3] = part_mask_resized * 255
    
    return rgba


def create_body_template(parts_dict, layout, cell_size=120, grid_rows=4, grid_cols=3):
    """
    Create a body template image with extracted parts placed anatomically.
    
    parts_dict: {part_id: rgba_image}
    layout: {part_id: (row, col)}
    """
    template = np.zeros((grid_rows * cell_size, grid_cols * cell_size, 3), dtype=np.uint8)
    template[:] = 50  # Dark gray background
    
    # Track which cells have content
    cells_filled = set()
    
    for part_id, rgba in parts_dict.items():
        if part_id not in layout:
            continue
        
        row, col = layout[part_id]
        
        # Resize part to fit cell
        ph, pw = rgba.shape[:2]
        if ph == 0 or pw == 0:
            continue
            
        scale = min(cell_size / pw, cell_size / ph) * 0.85
        new_w = max(1, int(pw * scale))
        new_h = max(1, int(ph * scale))
        
        part_resized = cv2.resize(rgba, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Position in cell
        y_start = row * cell_size + (cell_size - new_h) // 2
        x_start = col * cell_size + (cell_size - new_w) // 2
        
        # If cell already has content, offset slightly
        if (row, col) in cells_filled:
            x_start += 5
            y_start += 5
        cells_filled.add((row, col))
        
        # Blend using alpha
        y_end = min(y_start + new_h, template.shape[0])
        x_end = min(x_start + new_w, template.shape[1])
        
        actual_h = y_end - y_start
        actual_w = x_end - x_start
        
        if actual_h > 0 and actual_w > 0:
            alpha = part_resized[:actual_h, :actual_w, 3:4] / 255.0
            rgb = part_resized[:actual_h, :actual_w, :3]
            roi = template[y_start:y_end, x_start:x_end]
            template[y_start:y_end, x_start:x_end] = (
                alpha * rgb + (1 - alpha) * roi
            ).astype(np.uint8)
    
    return template


def create_figure(image_path, results, output_path):
    """Create the 4-panel visualization."""
    
    # Load image
    image = cv2.imread(str(image_path))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_h, img_w = image.shape[:2]
    
    # Get predictions
    result = results[0]
    boxes = result['pred_boxes_XYXY']
    if hasattr(boxes, 'cpu'):
        boxes = boxes.cpu().numpy()
    box = boxes[0].astype(int)
    x1, y1, x2, y2 = box
    
    dp = result['pred_densepose'][0]
    labels = dp.labels.cpu().numpy() if hasattr(dp.labels, 'cpu') else dp.labels
    pred_h, pred_w = labels.shape
    
    # Diagnostic: show exactly what's in the labels
    unique_parts = sorted([p for p in np.unique(labels) if p > 0])
    print(f"\n=== DIAGNOSTIC INFO ===")
    print(f"Image: {image_path}")
    print(f"Bounding box: [{x1}, {y1}, {x2}, {y2}]")
    print(f"Prediction shape: {labels.shape}")
    print(f"Unique part IDs detected: {unique_parts}")
    print(f"\nPart breakdown:")
    for p in unique_parts:
        count = np.sum(labels == p)
        name = PART_NAMES.get(p, 'Unknown')
        view = 'ANTERIOR' if p in ANTERIOR_PARTS else 'POSTERIOR'
        print(f"  Part {p:2d}: {name:30s} ({view}) - {count:6d} pixels")
    print(f"========================\n")
    
    # Extract all body parts
    all_parts = {}
    anterior_parts = {}
    posterior_parts = {}
    
    for part_id in unique_parts:
        rgba = extract_part_pixels(image_rgb, labels, box, part_id)
        if rgba is not None:
            all_parts[part_id] = rgba
            if part_id in ANTERIOR_PARTS:
                anterior_parts[part_id] = rgba
            elif part_id in POSTERIOR_PARTS:
                posterior_parts[part_id] = rgba
    
    print(f"\nExtracted {len(all_parts)} parts total")
    print(f"  Anterior: {len(anterior_parts)} parts - {sorted(anterior_parts.keys())}")
    print(f"  Posterior: {len(posterior_parts)} parts - {sorted(posterior_parts.keys())}")
    
    # Create full-image segmentation map
    box_h, box_w = y2 - y1, x2 - x1
    labels_resized = cv2.resize(labels.astype(np.float32), (box_w, box_h),
                                 interpolation=cv2.INTER_NEAREST).astype(np.uint8)
    
    I_full = np.zeros((img_h, img_w), dtype=np.uint8)
    y1c, y2c = max(0, y1), min(img_h, y2)
    x1c, x2c = max(0, x1), min(img_w, x2)
    I_full[y1c:y2c, x1c:x2c] = labels_resized[:y2c-y1c, :x2c-x1c]
    
    # Get colors
    colors = get_colors()
    cmap = ListedColormap(colors)
    
    # Create templates
    anterior_template = create_body_template(anterior_parts, ANTERIOR_LAYOUT)
    posterior_template = create_body_template(posterior_parts, POSTERIOR_LAYOUT)
    
    # Create figure - 2x2 layout
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # A: Original image with detection
    axes[0, 0].imshow(image_rgb)
    rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='lime', linewidth=2)
    axes[0, 0].add_patch(rect)
    axes[0, 0].set_title('A. Input Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # B: Body part segmentation with legend
    axes[0, 1].imshow(I_full, cmap=cmap, vmin=0, vmax=24)
    axes[0, 1].set_title('B. Body Part Segmentation', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Add legend
    legend_patches = []
    for part_id in unique_parts:
        color = colors[part_id]
        name = PART_NAMES.get(part_id, f'Part {part_id}')
        legend_patches.append(mpatches.Patch(color=color, label=f'{part_id}: {name}'))
    
    axes[0, 1].legend(handles=legend_patches, loc='center left', bbox_to_anchor=(1.02, 0.5),
                      fontsize=8, frameon=True, fancybox=True)
    
    # C: Anterior template
    axes[1, 0].imshow(anterior_template)
    axes[1, 0].set_title(f'C. Anterior (Front) - {len(anterior_parts)} parts', 
                         fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    # D: Posterior template  
    axes[1, 1].imshow(posterior_template)
    axes[1, 1].set_title(f'D. Posterior (Back) - {len(posterior_parts)} parts',
                         fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    
    # Main title
    fig.suptitle('DensePose: Body Surface Extraction from Single Image', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Caption
    caption = f"Extracted {len(all_parts)} body regions. This demonstrates coordinate-based surface mapping for coverage tracking."
    fig.text(0.5, 0.02, caption, ha='center', fontsize=11, style='italic')
    
    plt.tight_layout(rect=[0, 0.04, 0.85, 0.95])
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