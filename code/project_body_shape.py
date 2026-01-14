#!/usr/bin/env python3
"""
Project DensePose IUV output onto anatomically-shaped body templates.
Two views: ANTERIOR (front) and POSTERIOR (back), arranged like a body.
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

# DensePose body part definitions
# Format: part_id: (name, view, body_region)
PART_INFO = {
    1:  ("Torso", "back", "torso"),
    2:  ("Torso", "front", "torso"),
    3:  ("R Hand", "front", "r_hand"),
    4:  ("L Hand", "front", "l_hand"),
    5:  ("L Foot", "front", "l_foot"),
    6:  ("R Foot", "front", "r_foot"),
    7:  ("R Thigh", "back", "r_thigh"),
    8:  ("L Thigh", "back", "l_thigh"),
    9:  ("R Thigh", "front", "r_thigh"),
    10: ("L Thigh", "front", "l_thigh"),
    11: ("R Calf", "back", "r_calf"),
    12: ("L Calf", "back", "l_calf"),
    13: ("R Calf", "front", "r_calf"),
    14: ("L Calf", "front", "l_calf"),
    15: ("L Upper Arm", "front", "l_upper_arm"),
    16: ("R Upper Arm", "front", "r_upper_arm"),
    17: ("L Upper Arm", "back", "l_upper_arm"),
    18: ("R Upper Arm", "back", "r_upper_arm"),
    19: ("L Forearm", "front", "l_forearm"),
    20: ("R Forearm", "front", "r_forearm"),
    21: ("L Forearm", "back", "l_forearm"),
    22: ("R Forearm", "back", "r_forearm"),
    23: ("Head R", "front", "head"),  # Right side of head (visible from front)
    24: ("Head L", "front", "head"),  # Left side of head (visible from front)
}

# Which parts go in anterior vs posterior view
ANTERIOR_PARTS = [2, 3, 4, 5, 6, 9, 10, 13, 14, 15, 16, 19, 20, 23, 24]
POSTERIOR_PARTS = [1, 7, 8, 11, 12, 17, 18, 21, 22]

def create_body_template(width=400, height=700):
    """
    Create a body-shaped template with regions for each body part.
    Returns template image and a dict mapping part_id -> (x, y, w, h) regions.
    """
    template = np.full((height, width, 3), 30, dtype=np.uint8)
    
    # Define anatomical regions as (x, y, width, height) relative to template
    # Organized like a body: head at top, feet at bottom
    
    cx = width // 2  # Center x
    
    regions = {
        # Head - top center
        "head": (cx - 50, 10, 100, 80),
        
        # Torso - center
        "torso": (cx - 70, 100, 140, 180),
        
        # Upper arms - sides of torso
        "l_upper_arm": (cx + 70, 110, 50, 100),
        "r_upper_arm": (cx - 120, 110, 50, 100),
        
        # Forearms - below upper arms
        "l_forearm": (cx + 80, 210, 45, 100),
        "r_forearm": (cx - 125, 210, 45, 100),
        
        # Hands - end of arms
        "l_hand": (cx + 85, 310, 40, 50),
        "r_hand": (cx - 125, 310, 40, 50),
        
        # Thighs - below torso
        "l_thigh": (cx + 5, 290, 60, 130),
        "r_thigh": (cx - 65, 290, 60, 130),
        
        # Calves - below thighs
        "l_calf": (cx + 10, 420, 50, 130),
        "r_calf": (cx - 60, 420, 50, 130),
        
        # Feet - bottom
        "l_foot": (cx + 10, 550, 55, 60),
        "r_foot": (cx - 65, 550, 55, 60),
    }
    
    return template, regions


def project_to_body_template(image_rgb, labels, u_coords, v_coords, bbox, 
                              template_size=(400, 700), parts_to_include=None):
    """
    Project pixels onto a body-shaped template.
    
    Args:
        image_rgb: Original image
        labels: Body part labels (I), values 1-24
        u_coords, v_coords: UV coordinates (0-1)
        bbox: Bounding box [x1, y1, x2, y2]
        template_size: (width, height) of output template
        parts_to_include: List of part IDs to include (for anterior/posterior filtering)
    
    Returns:
        template: Body-shaped image with projected pixels
        coverage_info: Dict with coverage stats
    """
    width, height = template_size
    template, regions = create_body_template(width, height)
    pixel_count = np.zeros((height, width), dtype=np.int32)
    
    x1, y1, x2, y2 = bbox
    bbox_w = x2 - x1
    bbox_h = y2 - y1
    pred_h, pred_w = labels.shape
    
    scale_x = bbox_w / pred_w
    scale_y = bbox_h / pred_h
    
    img_h, img_w = image_rgb.shape[:2]
    
    for py in range(pred_h):
        for px in range(pred_w):
            part = labels[py, px]
            if part == 0:
                continue
            if parts_to_include is not None and part not in parts_to_include:
                continue
            
            # Get region for this part
            part_info = PART_INFO.get(part)
            if part_info is None:
                continue
            
            region_name = part_info[2]  # e.g., "torso", "head", etc.
            if region_name not in regions:
                continue
            
            rx, ry, rw, rh = regions[region_name]
            
            # Get UV coordinates
            u = u_coords[py, px]
            v = v_coords[py, px]
            
            # Map UV to position within this region
            template_x = int(rx + u * (rw - 1))
            template_y = int(ry + v * (rh - 1))
            
            # Bounds check
            if template_x < 0 or template_x >= width or template_y < 0 or template_y >= height:
                continue
            
            # Get pixel color from original image
            img_x = int(x1 + px * scale_x)
            img_y = int(y1 + py * scale_y)
            
            if img_x < 0 or img_x >= img_w or img_y < 0 or img_y >= img_h:
                continue
            
            rgb = image_rgb[img_y, img_x]
            
            # Paint pixel
            if pixel_count[template_y, template_x] == 0:
                template[template_y, template_x] = rgb
            else:
                # Average with existing
                old = template[template_y, template_x].astype(np.float32)
                n = pixel_count[template_y, template_x]
                new = (old * n + rgb.astype(np.float32)) / (n + 1)
                template[template_y, template_x] = new.astype(np.uint8)
            
            pixel_count[template_y, template_x] += 1
    
    # Fill small gaps
    filled_mask = (pixel_count > 0).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(filled_mask, kernel, iterations=2)
    
    # Calculate total region area for coverage
    total_region_area = 0
    for region_name, (rx, ry, rw, rh) in regions.items():
        total_region_area += rw * rh
    
    filled_pixels = np.sum(pixel_count > 0)
    coverage = filled_pixels / total_region_area * 100 if total_region_area > 0 else 0
    
    # Draw region outlines (faint)
    for region_name, (rx, ry, rw, rh) in regions.items():
        cv2.rectangle(template, (rx, ry), (rx + rw, ry + rh), (60, 60, 60), 1)
    
    return template, {"coverage": coverage, "filled_pixels": filled_pixels}


def main():
    print("="*60)
    print("DensePose Body-Shaped Projection")
    print("="*60)
    
    # Load image
    image = cv2.imread(str(IMAGE_PATH))
    if image is None:
        print(f"ERROR: Could not load {IMAGE_PATH}")
        return
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_h, img_w = image.shape[:2]
    print(f"Image size: {img_w} x {img_h}")
    
    # Load DensePose results
    results = torch.load(PKL_PATH, map_location='cpu', weights_only=False)
    result = results[0]
    
    # Get bounding box
    bbox = result['pred_boxes_XYXY'].cpu().numpy()[0].astype(int)
    x1, y1, x2, y2 = bbox
    print(f"Bounding box: [{x1}, {y1}, {x2}, {y2}]")
    
    # Get IUV predictions
    dp = result['pred_densepose'][0]
    labels = dp.labels.cpu().numpy()
    uv = dp.uv.cpu().numpy()
    u_coords = uv[0]
    v_coords = uv[1]
    
    print(f"Prediction size: {labels.shape[1]} x {labels.shape[0]}")
    print(f"Body pixels: {np.sum(labels > 0)}")
    
    # Count parts by view
    anterior_count = sum(np.sum(labels == p) for p in ANTERIOR_PARTS)
    posterior_count = sum(np.sum(labels == p) for p in POSTERIOR_PARTS)
    print(f"Anterior parts pixels: {anterior_count}")
    print(f"Posterior parts pixels: {posterior_count}")
    
    # Create anterior template
    print("\nProjecting ANTERIOR view...")
    anterior_template, anterior_info = project_to_body_template(
        image_rgb, labels, u_coords, v_coords, bbox,
        template_size=(400, 700),
        parts_to_include=ANTERIOR_PARTS
    )
    print(f"  Coverage: {anterior_info['coverage']:.1f}%")
    
    # Create posterior template
    print("Projecting POSTERIOR view...")
    posterior_template, posterior_info = project_to_body_template(
        image_rgb, labels, u_coords, v_coords, bbox,
        template_size=(400, 700),
        parts_to_include=POSTERIOR_PARTS
    )
    print(f"  Coverage: {posterior_info['coverage']:.1f}%")
    
    # Add labels
    cv2.putText(anterior_template, "ANTERIOR", (150, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    cv2.putText(anterior_template, f"{anterior_info['coverage']:.0f}% coverage", (140, 680),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    
    cv2.putText(posterior_template, "POSTERIOR", (140, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    cv2.putText(posterior_template, f"{posterior_info['coverage']:.0f}% coverage", (140, 680),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    
    # Create combined figure
    # Panel A: Original image with bbox
    panel_a = image_rgb.copy()
    cv2.rectangle(panel_a, (x1, y1), (x2, y2), (0, 255, 0), 3)
    
    # Scale panel A to match template height
    scale = 700 / panel_a.shape[0]
    panel_a_scaled = cv2.resize(panel_a, None, fx=scale, fy=scale)
    
    # Add label to panel A
    cv2.putText(panel_a_scaled, "INPUT", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Combine: Input | Anterior | Posterior
    spacer = np.full((700, 15, 3), 30, dtype=np.uint8)
    
    figure = np.hstack([
        panel_a_scaled,
        spacer,
        anterior_template,
        spacer,
        posterior_template
    ])
    
    # Save
    output_path = OUTPUT_DIR / "body_projection_demo.png"
    cv2.imwrite(str(output_path), cv2.cvtColor(figure, cv2.COLOR_RGB2BGR))
    print(f"\nSaved: {output_path}")
    
    # Also save individual templates
    cv2.imwrite(str(OUTPUT_DIR / "anterior_template.png"), 
                cv2.cvtColor(anterior_template, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(OUTPUT_DIR / "posterior_template.png"), 
                cv2.cvtColor(posterior_template, cv2.COLOR_RGB2BGR))
    
    print(f"Saved: {OUTPUT_DIR / 'anterior_template.png'}")
    print(f"Saved: {OUTPUT_DIR / 'posterior_template.png'}")
    
    # Print per-region coverage
    print(f"\n{'Region':<20} {'Pixels':>8}")
    print("-" * 30)
    for part_id in sorted(PART_INFO.keys()):
        name, view, region = PART_INFO[part_id]
        count = np.sum(labels == part_id)
        if count > 0:
            print(f"{part_id:2d}: {name:<15} ({view:<5}) {count:>6}")
    
    print("\n" + "="*60)
    print("DONE")
    print("="*60)


if __name__ == "__main__":
    main()
