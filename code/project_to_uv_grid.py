#!/usr/bin/env python3
"""
Project DensePose IUV output onto a universal UV coordinate grid.

This demonstrates the "genome alignment" concept:
- Each body part (I=1-24) is like a chromosome
- U, V coordinates (0-1) are positions within that chromosome
- Pixel appearance values are projected onto their universal coordinates
"""

import pickle
import numpy as np
import cv2
from pathlib import Path
import sys

# Configuration
PROJECT_ROOT = Path("/nfs/turbo/lsa-tlasisi1/tlasisi/Leakey2025")
IMAGE_PATH = PROJECT_ROOT / "data" / "chimp_test.jpg"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Find the pkl file
PKL_CANDIDATES = [
    OUTPUT_DIR / "densepose_iuv_results.pkl",
    OUTPUT_DIR / "densepose_results.pkl", 
    PROJECT_ROOT / "densepose_iuv_results.pkl",
]

def find_pkl_file():
    """Find the DensePose results pkl file."""
    for path in PKL_CANDIDATES:
        if path.exists():
            print(f"Found pkl file: {path}")
            return path
    
    # If not found, search
    print("Searching for pkl files...")
    for pkl in OUTPUT_DIR.glob("*.pkl"):
        print(f"  Found: {pkl}")
    for pkl in PROJECT_ROOT.glob("**/*.pkl"):
        print(f"  Found: {pkl}")
    
    sys.exit("ERROR: Could not find DensePose pkl file. Check paths above.")

def project_to_grid(image, labels, u_coords, v_coords, bbox):
    """
    Project image pixels onto the UV coordinate grid.
    
    Args:
        image: Original RGB image
        labels: Body part labels (I), shape (H, W), values 1-24
        u_coords: U coordinates, shape (H, W), values 0-1
        v_coords: V coordinates, shape (H, W), values 0-1
        bbox: Bounding box [x1, y1, x2, y2]
    
    Returns:
        template: 800x1200 RGB image (4 rows x 6 cols of 200x200 cells)
        coverage: Percentage of template filled
    """
    # Template: 4 rows x 6 cols = 24 body parts, each 200x200
    template = np.full((800, 1200, 3), 40, dtype=np.uint8)  # Dark gray background
    coverage_mask = np.zeros((800, 1200), dtype=bool)
    
    # Draw grid lines
    for i in range(1, 6):  # Vertical lines
        cv2.line(template, (i * 200, 0), (i * 200, 800), (80, 80, 80), 1)
    for i in range(1, 4):  # Horizontal lines
        cv2.line(template, (0, i * 200), (1200, i * 200), (80, 80, 80), 1)
    
    x1, y1, x2, y2 = bbox
    pred_h, pred_w = labels.shape
    
    # Scale factors from prediction size to bbox size
    bbox_h = y2 - y1
    bbox_w = x2 - x1
    scale_y = bbox_h / pred_h
    scale_x = bbox_w / pred_w
    
    pixels_projected = 0
    
    for py in range(pred_h):
        for px in range(pred_w):
            part = labels[py, px]
            if part == 0:  # Background
                continue
            
            # Get UV coordinates
            u = u_coords[py, px]
            v = v_coords[py, px]
            
            # Get pixel color from original image
            img_x = int(x1 + px * scale_x)
            img_y = int(y1 + py * scale_y)
            
            if 0 <= img_y < image.shape[0] and 0 <= img_x < image.shape[1]:
                rgb = image[img_y, img_x]
            else:
                continue
            
            # Calculate position in template grid
            # Part 1-24 maps to cells in row-major order
            cell_idx = part - 1
            cell_row = cell_idx // 6  # 0-3
            cell_col = cell_idx % 6   # 0-5
            
            # U, V give position within the 200x200 cell
            template_x = int(cell_col * 200 + u * 199)
            template_y = int(cell_row * 200 + v * 199)
            
            # Clamp to valid range
            template_x = max(0, min(1199, template_x))
            template_y = max(0, min(799, template_y))
            
            # Paint pixel (use small circle to fill gaps)
            cv2.circle(template, (template_x, template_y), 2, rgb.tolist(), -1)
            coverage_mask[max(0,template_y-2):min(800,template_y+3), 
                         max(0,template_x-2):min(1200,template_x+3)] = True
            pixels_projected += 1
    
    # Calculate coverage (approximate - count non-background pixels)
    coverage = np.sum(coverage_mask) / (800 * 1200) * 100
    
    print(f"Pixels projected: {pixels_projected}")
    print(f"Coverage: {coverage:.1f}%")
    
    return template, coverage

def create_demo_figure(image_path, pkl_path, output_path):
    """Create the 3-panel demo figure."""
    
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        sys.exit(f"ERROR: Could not load image: {image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f"Loaded image: {image.shape}")
    
    # Load DensePose results
    with open(pkl_path, 'rb') as f:
        results = pickle.load(f)
    
    result = results[0]
    
    # Extract bounding box
    boxes = result['pred_boxes_XYXY']
    if hasattr(boxes, 'cpu'):
        boxes = boxes.cpu().numpy()
    bbox = boxes[0].astype(int)
    x1, y1, x2, y2 = bbox
    print(f"Bounding box: [{x1}, {y1}, {x2}, {y2}]")
    
    # Extract IUV data
    dp = result['pred_densepose'][0]
    
    if hasattr(dp, 'labels'):
        # IUV format
        labels = dp.labels.cpu().numpy() if hasattr(dp.labels, 'cpu') else dp.labels
        uv = dp.uv.cpu().numpy() if hasattr(dp.uv, 'cpu') else dp.uv
        u_coords = uv[0]
        v_coords = uv[1]
        print(f"IUV format - Labels shape: {labels.shape}, UV shape: {uv.shape}")
        print(f"Unique parts: {np.unique(labels)}")
    else:
        sys.exit("ERROR: pkl doesn't contain IUV labels. May be CSE format.")
    
    # Project to grid
    template, coverage = project_to_grid(image_rgb, labels, u_coords, v_coords, bbox)
    
    # Create figure
    # Panel A: Original image with bbox (scaled to ~400px height)
    scale = 400 / image.shape[0]
    panel_a = cv2.resize(image_rgb, None, fx=scale, fy=scale)
    # Draw bbox
    bx1, by1 = int(x1 * scale), int(y1 * scale)
    bx2, by2 = int(x2 * scale), int(y2 * scale)
    cv2.rectangle(panel_a, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
    
    # Panel C: The projection (scale to match height)
    panel_c = cv2.resize(template, (600, 400))
    
    # Add part numbers to Panel C
    for part in range(1, 25):
        cell_idx = part - 1
        cell_row = cell_idx // 6
        cell_col = cell_idx % 6
        cx = int((cell_col * 200 + 100) * 600 / 1200)
        cy = int((cell_row * 200 + 20) * 400 / 800)
        cv2.putText(panel_c, str(part), (cx-10, cy), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    
    # Combine panels side by side
    # Make heights match
    h = 400
    panel_a_resized = cv2.resize(panel_a, (int(panel_a.shape[1] * h / panel_a.shape[0]), h))
    
    # Add labels
    cv2.putText(panel_a_resized, "A. Input", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(panel_c, f"B. UV Projection ({coverage:.1f}% coverage)", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Concatenate
    figure = np.hstack([panel_a_resized, np.ones((h, 20, 3), dtype=np.uint8) * 40, panel_c])
    
    # Save
    output_bgr = cv2.cvtColor(figure, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), output_bgr)
    print(f"\nSaved: {output_path}")
    
    # Also save full-resolution template
    template_path = output_path.parent / "uv_template_fullres.png"
    cv2.imwrite(str(template_path), cv2.cvtColor(template, cv2.COLOR_RGB2BGR))
    print(f"Saved full-res template: {template_path}")

def main():
    print("="*60)
    print("DensePose UV Projection Demo")
    print("="*60)
    
    # Find pkl file
    pkl_path = find_pkl_file()
    
    # Check image exists
    if not IMAGE_PATH.exists():
        sys.exit(f"ERROR: Image not found: {IMAGE_PATH}")
    
    # Create output
    output_path = OUTPUT_DIR / "uv_projection_demo.png"
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    create_demo_figure(IMAGE_PATH, pkl_path, output_path)
    
    print("\n" + "="*60)
    print("DONE!")
    print("="*60)

if __name__ == "__main__":
    main()
