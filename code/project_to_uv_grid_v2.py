#!/usr/bin/env python3
"""
Project DensePose IUV output onto a universal UV coordinate grid.
Version 2: Fixed based on actual data structure diagnostic.
"""

import numpy as np
import cv2
from pathlib import Path
import torch

# Configuration
PROJECT_ROOT = Path("/nfs/turbo/lsa-tlasisi1/tlasisi/Leakey2025")
IMAGE_PATH = PROJECT_ROOT / "data" / "chimp_test.jpg"
OUTPUT_DIR = PROJECT_ROOT / "output"
PKL_PATH = OUTPUT_DIR / "densepose_iuv_results.pkl"

def main():
    print("="*60)
    print("DensePose UV Projection - Version 2")
    print("="*60)
    
    # Load image
    image = cv2.imread(str(IMAGE_PATH))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_h, img_w = image.shape[:2]
    print(f"Image size: {img_w} x {img_h}")
    
    # Load DensePose results
    results = torch.load(PKL_PATH, map_location='cpu', weights_only=False)
    result = results[0]
    
    # Get bounding box
    bbox = result['pred_boxes_XYXY'].cpu().numpy()[0]
    x1, y1, x2, y2 = bbox.astype(int)
    bbox_w = x2 - x1
    bbox_h = y2 - y1
    print(f"Bounding box: [{x1}, {y1}, {x2}, {y2}] ({bbox_w} x {bbox_h})")
    
    # Get IUV predictions
    dp = result['pred_densepose'][0]
    labels = dp.labels.cpu().numpy()  # Shape: (pred_h, pred_w), values 0-24
    uv = dp.uv.cpu().numpy()          # Shape: (2, pred_h, pred_w)
    u_coords = uv[0]
    v_coords = uv[1]
    
    pred_h, pred_w = labels.shape
    print(f"Prediction size: {pred_w} x {pred_h}")
    
    # Count body pixels
    body_mask = labels > 0
    n_body_pixels = np.sum(body_mask)
    print(f"Body pixels: {n_body_pixels} ({n_body_pixels/(pred_h*pred_w)*100:.1f}%)")
    
    # Create template: 4 rows x 6 cols = 24 parts, each 200x200
    # But let's make it higher resolution for better coverage: 400x400 per cell
    CELL_SIZE = 200
    ROWS = 4
    COLS = 6
    template_h = ROWS * CELL_SIZE
    template_w = COLS * CELL_SIZE
    
    template = np.full((template_h, template_w, 3), 30, dtype=np.uint8)  # Dark gray
    pixel_count = np.zeros((template_h, template_w), dtype=np.int32)  # For averaging
    
    # Scale factors: prediction grid -> bounding box in image
    scale_x = bbox_w / pred_w
    scale_y = bbox_h / pred_h
    
    print(f"\nProjecting pixels to template...")
    
    # Iterate through all prediction pixels
    for py in range(pred_h):
        for px in range(pred_w):
            part = labels[py, px]
            if part == 0:  # Background
                continue
            
            # Get UV coordinates for this pixel
            u = u_coords[py, px]
            v = v_coords[py, px]
            
            # Get corresponding image pixel location
            img_x = int(x1 + px * scale_x)
            img_y = int(y1 + py * scale_y)
            
            # Bounds check
            if img_x < 0 or img_x >= img_w or img_y < 0 or img_y >= img_h:
                continue
            
            # Get RGB from original image
            rgb = image_rgb[img_y, img_x].astype(np.int32)
            
            # Calculate template position
            # Part 1-24 maps to cell (row, col) in row-major order
            cell_idx = part - 1  # 0-23
            cell_row = cell_idx // COLS  # 0-3
            cell_col = cell_idx % COLS   # 0-5
            
            # U, V give position within cell (0-1 -> 0 to CELL_SIZE-1)
            template_x = int(cell_col * CELL_SIZE + u * (CELL_SIZE - 1))
            template_y = int(cell_row * CELL_SIZE + v * (CELL_SIZE - 1))
            
            # Clamp to valid range
            template_x = max(0, min(template_w - 1, template_x))
            template_y = max(0, min(template_h - 1, template_y))
            
            # Accumulate color (for averaging overlapping pixels)
            if pixel_count[template_y, template_x] == 0:
                template[template_y, template_x] = rgb.astype(np.uint8)
            else:
                # Running average
                old_rgb = template[template_y, template_x].astype(np.int32)
                n = pixel_count[template_y, template_x]
                new_rgb = ((old_rgb * n + rgb) / (n + 1)).astype(np.uint8)
                template[template_y, template_x] = new_rgb
            
            pixel_count[template_y, template_x] += 1
    
    # Calculate coverage
    filled_pixels = np.sum(pixel_count > 0)
    coverage = filled_pixels / (template_h * template_w) * 100
    print(f"Template pixels filled: {filled_pixels} / {template_h * template_w} ({coverage:.1f}%)")
    
    # Fill small gaps using dilation
    print("Filling small gaps...")
    filled_mask = (pixel_count > 0).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    dilated_mask = cv2.dilate(filled_mask, kernel, iterations=2)
    
    # For gap pixels, use nearest neighbor interpolation
    template_filled = template.copy()
    gap_pixels = (dilated_mask > 0) & (pixel_count == 0)
    
    # Simple gap filling: for each gap pixel, find nearest filled pixel
    gap_coords = np.where(gap_pixels)
    filled_coords = np.where(pixel_count > 0)
    
    if len(gap_coords[0]) > 0 and len(filled_coords[0]) > 0:
        from scipy.spatial import cKDTree
        filled_points = np.column_stack(filled_coords)
        tree = cKDTree(filled_points)
        gap_points = np.column_stack(gap_coords)
        _, indices = tree.query(gap_points, k=1)
        
        for i, (gy, gx) in enumerate(zip(gap_coords[0], gap_coords[1])):
            nearest_idx = indices[i]
            ny, nx = filled_coords[0][nearest_idx], filled_coords[1][nearest_idx]
            template_filled[gy, gx] = template[ny, nx]
    
    coverage_after_fill = np.sum(dilated_mask > 0) / (template_h * template_w) * 100
    print(f"Coverage after gap fill: {coverage_after_fill:.1f}%")
    
    # Draw grid lines
    for i in range(1, COLS):
        cv2.line(template_filled, (i * CELL_SIZE, 0), (i * CELL_SIZE, template_h), (60, 60, 60), 1)
    for i in range(1, ROWS):
        cv2.line(template_filled, (0, i * CELL_SIZE), (template_w, i * CELL_SIZE), (60, 60, 60), 1)
    
    # Add part numbers
    for part in range(1, 25):
        cell_idx = part - 1
        cell_row = cell_idx // COLS
        cell_col = cell_idx % COLS
        cx = cell_col * CELL_SIZE + 10
        cy = cell_row * CELL_SIZE + 25
        cv2.putText(template_filled, str(part), (cx, cy),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
    
    # Create figure with panels
    # Panel A: Original image with bbox
    panel_a = image_rgb.copy()
    cv2.rectangle(panel_a, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.putText(panel_a, "A. Input Image", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Panel B: UV Projection
    panel_b = template_filled.copy()
    cv2.putText(panel_b, f"B. UV Coordinate Projection ({coverage_after_fill:.0f}% coverage)", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Scale panels to similar height
    target_h = 500
    scale_a = target_h / panel_a.shape[0]
    scale_b = target_h / panel_b.shape[0]
    
    panel_a_scaled = cv2.resize(panel_a, None, fx=scale_a, fy=scale_a)
    panel_b_scaled = cv2.resize(panel_b, None, fx=scale_b, fy=scale_b)
    
    # Add spacer and combine
    spacer = np.full((target_h, 20, 3), 30, dtype=np.uint8)
    figure = np.hstack([panel_a_scaled, spacer, panel_b_scaled])
    
    # Save outputs
    output_figure = OUTPUT_DIR / "uv_projection_demo.png"
    output_template = OUTPUT_DIR / "uv_template_fullres.png"
    
    cv2.imwrite(str(output_figure), cv2.cvtColor(figure, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(output_template), cv2.cvtColor(template_filled, cv2.COLOR_RGB2BGR))
    
    print(f"\nSaved: {output_figure}")
    print(f"Saved: {output_template}")
    
    # Print part coverage breakdown
    print(f"\nPer-part coverage:")
    for part in range(1, 25):
        cell_idx = part - 1
        cell_row = cell_idx // COLS
        cell_col = cell_idx % COLS
        cell_slice = pixel_count[cell_row*CELL_SIZE:(cell_row+1)*CELL_SIZE,
                                 cell_col*CELL_SIZE:(cell_col+1)*CELL_SIZE]
        part_coverage = np.sum(cell_slice > 0) / (CELL_SIZE * CELL_SIZE) * 100
        if part_coverage > 0:
            print(f"  Part {part:2d}: {part_coverage:5.1f}%")
    
    print("\n" + "="*60)
    print("DONE")
    print("="*60)

if __name__ == "__main__":
    main()
