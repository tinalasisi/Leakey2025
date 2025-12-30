#!/usr/bin/env python3
"""
Project DensePose IUV predictions onto a texture atlas.
Shows how pixels map to a standardized body surface coordinate system.
"""

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
PROJECT = Path("/nfs/turbo/lsa-tlasisi1/tlasisi/Leakey2025")
PKL_PATH = PROJECT / "output" / "densepose_iuv_results.pkl"
IMG_PATH = PROJECT / "data" / "chimp_test.jpg"
OUT_PATH = PROJECT / "output" / "texture_atlas_demo.png"

# Atlas layout: 4 rows × 6 cols = 24 body parts, each cell 200×200
CELL_SIZE = 200
ROWS, COLS = 4, 6

# Load data
data = torch.load(PKL_PATH, map_location='cpu', weights_only=False)
image = cv2.cvtColor(cv2.imread(str(IMG_PATH)), cv2.COLOR_BGR2RGB)
img_h, img_w = image.shape[:2]

# Extract predictions
result = data[0]
box = result['pred_boxes_XYXY'][0].numpy().astype(int)
x1, y1, x2, y2 = box

dp = result['pred_densepose'][0]
I = dp.labels.cpu().numpy()  # body part indices
U = dp.uv[0].cpu().numpy()   # U coordinates (0-1)
V = dp.uv[1].cpu().numpy()   # V coordinates (0-1)

# Resize predictions to bounding box size
box_h, box_w = y2 - y1, x2 - x1
I_box = cv2.resize(I.astype(np.float32), (box_w, box_h), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
U_box = cv2.resize(U, (box_w, box_h), interpolation=cv2.INTER_LINEAR)
V_box = cv2.resize(V, (box_w, box_h), interpolation=cv2.INTER_LINEAR)

# Create blank atlas
atlas = np.zeros((ROWS * CELL_SIZE, COLS * CELL_SIZE, 3), dtype=np.uint8)
coverage_mask = np.zeros((ROWS * CELL_SIZE, COLS * CELL_SIZE), dtype=bool)

# Map pixels to atlas
for y in range(box_h):
    for x in range(box_w):
        part = int(I_box[y, x])
        if part == 0 or part > 24:
            continue
        
        # Get image pixel color
        img_y, img_x = y1 + y, x1 + x
        if 0 <= img_y < img_h and 0 <= img_x < img_w:
            color = image[img_y, img_x]
        else:
            continue
        
        # Map part (1-24) to atlas cell
        cell_idx = part - 1
        cell_row, cell_col = cell_idx // COLS, cell_idx % COLS
        
        # Map U,V to position within cell
        u, v = U_box[y, x], V_box[y, x]
        atlas_x = int(cell_col * CELL_SIZE + u * (CELL_SIZE - 1))
        atlas_y = int(cell_row * CELL_SIZE + v * (CELL_SIZE - 1))
        
        # Place pixel
        atlas[atlas_y, atlas_x] = color
        coverage_mask[atlas_y, atlas_x] = True

# Calculate coverage
coverage_pct = 100 * coverage_mask.sum() / coverage_mask.size

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Input image with bounding box
axes[0].imshow(image)
rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='lime', linewidth=2)
axes[0].add_patch(rect)
axes[0].set_title('Input Image', fontsize=14)
axes[0].axis('off')

# Texture atlas with grid
axes[1].imshow(atlas)
for i in range(1, ROWS):
    axes[1].axhline(i * CELL_SIZE, color='white', linewidth=0.5, alpha=0.5)
for j in range(1, COLS):
    axes[1].axvline(j * CELL_SIZE, color='white', linewidth=0.5, alpha=0.5)
axes[1].set_title(f'Texture Atlas ({coverage_pct:.1f}% coverage)', fontsize=14)
axes[1].axis('off')

plt.suptitle('DensePose: Pixel → Body Surface Mapping', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150, bbox_inches='tight', facecolor='white')
print(f"Saved to {OUT_PATH}")
print(f"Coverage: {coverage_pct:.1f}%")
