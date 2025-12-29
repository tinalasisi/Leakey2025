#!/usr/bin/env python3
"""
TPS Phenomic Mapping Demo
=========================
Demonstrates that TPS produces a HEATMAP on a template, not a "warped photo."
This directly addresses Reviewer 2's "visual chaos" concern.

Usage:
    python tps_demo.py --image_dir OpenApePose/images --ann_file OpenApePose/annotations/oap_all.json

Output:
    tps_demo_figure.png - 4-panel figure for rebuttal
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.interpolate import RBFInterpolator
from PIL import Image
from pathlib import Path

# Landmark definitions from OpenApePose
LANDMARK_NAMES = [
    'nose', 'left_eye', 'right_eye', 'head_top', 'neck',
    'left_shoulder', 'left_elbow', 'left_wrist',
    'right_shoulder', 'right_elbow', 'right_wrist',
    'pelvis', 'left_knee', 'left_ankle', 'right_knee', 'right_ankle'
]

# Define a canonical TEMPLATE - where landmarks "should" be on a standardized body
# This is an anterior view template, normalized to [0, 1] space
TEMPLATE_LANDMARKS = np.array([
    [0.50, 0.08],  # 0: nose (center top)
    [0.45, 0.06],  # 1: left_eye
    [0.55, 0.06],  # 2: right_eye
    [0.50, 0.02],  # 3: head_top
    [0.50, 0.15],  # 4: neck
    [0.35, 0.22],  # 5: left_shoulder
    [0.25, 0.38],  # 6: left_elbow
    [0.18, 0.52],  # 7: left_wrist
    [0.65, 0.22],  # 8: right_shoulder
    [0.75, 0.38],  # 9: right_elbow
    [0.82, 0.52],  # 10: right_wrist
    [0.50, 0.50],  # 11: pelvis (center)
    [0.38, 0.70],  # 12: left_knee
    [0.35, 0.92],  # 13: left_ankle
    [0.62, 0.70],  # 14: right_knee
    [0.65, 0.92],  # 15: right_ankle
])


def load_annotations(ann_file):
    """Load OpenApePose annotations."""
    with open(ann_file) as f:
        data = json.load(f)
    # Handle both list format and dict with 'data' key
    if isinstance(data, dict) and 'data' in data:
        return data['data']
    return data


def find_good_example(annotations, species_filter=None, min_visible=14):
    """Find an image with high landmark visibility."""
    for ann in annotations:
        if species_filter and ann['species'].lower() != species_filter.lower():
            continue
        visibility = ann['visibility']
        if sum(visibility) >= min_visible:
            return ann
    return None


def parse_landmarks(ann):
    """Parse landmark coordinates from annotation."""
    landmarks = np.array(ann['landmarks']).reshape(-1, 2)
    visibility = np.array(ann['visibility'])
    return landmarks, visibility


def compute_tps_mapping(source_points, target_points):
    """
    Compute TPS interpolation from source to target coordinates.
    
    This creates a function that maps ANY point in source space
    to its corresponding point in target space.
    """
    # RBFInterpolator with thin_plate_spline kernel
    # We need to map source -> target for each coordinate
    rbf_x = RBFInterpolator(source_points, target_points[:, 0], kernel='thin_plate_spline')
    rbf_y = RBFInterpolator(source_points, target_points[:, 1], kernel='thin_plate_spline')
    
    def mapping_function(points):
        """Map points from source space to target space."""
        x_mapped = rbf_x(points)
        y_mapped = rbf_y(points)
        return np.column_stack([x_mapped, y_mapped])
    
    return mapping_function


def sample_body_region(image, landmarks, visibility, n_samples=500):
    """
    Sample pixels from the body region defined by visible landmarks.
    Returns pixel coordinates and their RGB values.
    """
    # Get visible landmarks
    visible_idx = visibility > 0
    visible_landmarks = landmarks[visible_idx]
    
    # Create a simple bounding region (convex hull would be better but this is a demo)
    min_x, min_y = visible_landmarks.min(axis=0)
    max_x, max_y = visible_landmarks.max(axis=0)
    
    # Add some padding
    pad_x = (max_x - min_x) * 0.1
    pad_y = (max_y - min_y) * 0.1
    min_x, min_y = max(0, min_x - pad_x), max(0, min_y - pad_y)
    max_x = min(image.shape[1], max_x + pad_x)
    max_y = min(image.shape[0], max_y + pad_y)
    
    # Sample random points within the bounding box
    np.random.seed(42)  # Reproducibility
    sample_x = np.random.uniform(min_x, max_x, n_samples)
    sample_y = np.random.uniform(min_y, max_y, n_samples)
    sample_points = np.column_stack([sample_x, sample_y])
    
    # Get RGB values at these points
    pixel_x = np.clip(sample_x.astype(int), 0, image.shape[1] - 1)
    pixel_y = np.clip(sample_y.astype(int), 0, image.shape[0] - 1)
    rgb_values = image[pixel_y, pixel_x, :]
    
    return sample_points, rgb_values


def create_template_heatmap(template_coords, rgb_values, grid_resolution=50):
    """
    Create a heatmap on the template by aggregating RGB values.
    This is the KEY output - showing that we get a data visualization,
    not a "warped photo."
    """
    # Create grid on template
    grid_x = np.linspace(0, 1, grid_resolution)
    grid_y = np.linspace(0, 1, grid_resolution)
    
    # Initialize accumulators
    color_sum = np.zeros((grid_resolution, grid_resolution, 3))
    count = np.zeros((grid_resolution, grid_resolution))
    
    # Bin each sample point into the grid
    for (tx, ty), rgb in zip(template_coords, rgb_values):
        if 0 <= tx <= 1 and 0 <= ty <= 1:
            gx = int(tx * (grid_resolution - 1))
            gy = int(ty * (grid_resolution - 1))
            color_sum[gy, gx, :] += rgb
            count[gy, gx] += 1
    
    # Average where we have samples
    with np.errstate(divide='ignore', invalid='ignore'):
        avg_color = color_sum / count[:, :, np.newaxis]
        avg_color = np.nan_to_num(avg_color, nan=255)  # White for no data
    
    # Also return coverage map
    coverage = count > 0
    
    return avg_color.astype(np.uint8), coverage, count


def create_demo_figure(image, landmarks, visibility, template_coords, 
                       rgb_values, heatmap, coverage, output_path):
    """Create the 4-panel demo figure for the rebuttal."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Panel A: Original image with landmarks
    ax = axes[0, 0]
    ax.imshow(image)
    visible_idx = visibility > 0
    ax.scatter(landmarks[visible_idx, 0], landmarks[visible_idx, 1], 
               c='lime', s=100, edgecolors='black', linewidths=2, zorder=5)
    ax.scatter(landmarks[~visible_idx, 0], landmarks[~visible_idx, 1], 
               c='red', s=100, edgecolors='black', linewidths=2, marker='x', zorder=5)
    # Draw skeleton
    skeleton = [(0, 4), (4, 5), (4, 8), (5, 6), (6, 7), (8, 9), (9, 10),
                (4, 11), (11, 12), (12, 13), (11, 14), (14, 15)]
    for i, j in skeleton:
        if visibility[i] and visibility[j]:
            ax.plot([landmarks[i, 0], landmarks[j, 0]], 
                   [landmarks[i, 1], landmarks[j, 1]], 'c-', linewidth=2)
    ax.set_title('A. Original Image + Detected Landmarks', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Panel B: Template with mapped sample points
    ax = axes[0, 1]
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(1.1, -0.1)  # Flip y for image coordinates
    # Draw template landmarks
    ax.scatter(TEMPLATE_LANDMARKS[:, 0], TEMPLATE_LANDMARKS[:, 1], 
               c='black', s=150, marker='o', zorder=5, label='Template landmarks')
    # Draw template skeleton
    for i, j in skeleton:
        ax.plot([TEMPLATE_LANDMARKS[i, 0], TEMPLATE_LANDMARKS[j, 0]], 
               [TEMPLATE_LANDMARKS[i, 1], TEMPLATE_LANDMARKS[j, 1]], 'k-', linewidth=2)
    # Show mapped sample points colored by their RGB
    colors = rgb_values / 255.0
    ax.scatter(template_coords[:, 0], template_coords[:, 1], 
               c=colors, s=10, alpha=0.5)
    ax.set_title('B. Sample Points Mapped to Template\n(via TPS coordinate transformation)', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Template X')
    ax.set_ylabel('Template Y')
    ax.set_aspect('equal')
    
    # Panel C: Color heatmap on template
    ax = axes[1, 0]
    ax.imshow(heatmap, extent=[0, 1, 1, 0])
    ax.scatter(TEMPLATE_LANDMARKS[:, 0], TEMPLATE_LANDMARKS[:, 1], 
               c='black', s=100, marker='o', edgecolors='white', linewidths=2, zorder=5)
    for i, j in skeleton:
        ax.plot([TEMPLATE_LANDMARKS[i, 0], TEMPLATE_LANDMARKS[j, 0]], 
               [TEMPLATE_LANDMARKS[i, 1], TEMPLATE_LANDMARKS[j, 1]], 'k-', linewidth=1)
    ax.set_title('C. Phenomic Map: Average Color at Each Template Location\n(THIS is the output, not a "warped photo")', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Template X')
    ax.set_ylabel('Template Y')
    
    # Panel D: Coverage map
    ax = axes[1, 1]
    im = ax.imshow(coverage.astype(float), extent=[0, 1, 1, 0], cmap='Greens', vmin=0, vmax=1)
    ax.scatter(TEMPLATE_LANDMARKS[:, 0], TEMPLATE_LANDMARKS[:, 1], 
               c='black', s=100, marker='o', edgecolors='white', linewidths=2, zorder=5)
    for i, j in skeleton:
        ax.plot([TEMPLATE_LANDMARKS[i, 0], TEMPLATE_LANDMARKS[j, 0]], 
               [TEMPLATE_LANDMARKS[i, 1], TEMPLATE_LANDMARKS[j, 1]], 'k-', linewidth=1)
    ax.set_title('D. Coverage Map: Which Template Regions Were Observed\n(Green = observed, White = not visible in this image)', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Template X')
    ax.set_ylabel('Template Y')
    plt.colorbar(im, ax=ax, label='Observed')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved demo figure to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='TPS Phenomic Mapping Demo')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Path to OpenApePose images directory')
    parser.add_argument('--ann_file', type=str, required=True,
                        help='Path to annotations JSON file')
    parser.add_argument('--species', type=str, default='Siamang',
                        help='Species to filter for (default: Siamang)')
    parser.add_argument('--output', type=str, default='tps_demo_figure.png',
                        help='Output figure path')
    args = parser.parse_args()
    
    # Load annotations
    print("Loading annotations...")
    annotations = load_annotations(args.ann_file)
    print(f"Loaded {len(annotations)} annotations")
    
    # Find a good example
    print(f"Finding a {args.species} image with good landmark visibility...")
    example = find_good_example(annotations, species_filter=args.species, min_visible=15)
    if example is None:
        print(f"No suitable {args.species} image found, trying any species...")
        example = find_good_example(annotations, min_visible=15)
    
    if example is None:
        print("ERROR: No suitable image found!")
        return
    
    print(f"Using: {example['file']} ({example['species']})")
    print(f"  Visible landmarks: {sum(example['visibility'])}/16")
    
    # Load image
    image_path = Path(args.image_dir) / example['file']
    image = np.array(Image.open(image_path))
    print(f"  Image size: {image.shape}")
    
    # Parse landmarks
    landmarks, visibility = parse_landmarks(example)
    
    # Normalize landmarks to [0, 1] for TPS
    # Use only visible landmarks for the mapping
    visible_idx = visibility > 0
    source_landmarks = landmarks[visible_idx]
    target_landmarks = TEMPLATE_LANDMARKS[visible_idx]
    
    # Normalize source landmarks to approximate [0, 1] range
    img_h, img_w = image.shape[:2]
    source_normalized = source_landmarks.copy()
    source_normalized[:, 0] /= img_w
    source_normalized[:, 1] /= img_h
    
    # Compute TPS mapping
    print("Computing TPS mapping...")
    tps_map = compute_tps_mapping(source_normalized, target_landmarks)
    
    # Sample body pixels
    print("Sampling body region...")
    sample_points, rgb_values = sample_body_region(image, landmarks, visibility, n_samples=2000)
    
    # Normalize sample points
    sample_normalized = sample_points.copy()
    sample_normalized[:, 0] /= img_w
    sample_normalized[:, 1] /= img_h
    
    # Map sample points to template coordinates
    print("Mapping samples to template...")
    template_coords = tps_map(sample_normalized)
    
    # Create heatmap
    print("Creating phenomic heatmap...")
    heatmap, coverage, count = create_template_heatmap(template_coords, rgb_values, grid_resolution=30)
    
    # Create figure
    print("Generating figure...")
    create_demo_figure(image, landmarks, visibility, template_coords,
                       rgb_values, heatmap, coverage, args.output)
    
    print("\n" + "="*60)
    print("DEMO COMPLETE")
    print("="*60)
    print("""
This figure demonstrates that:
    
1. TPS is a COORDINATE MAPPING, not image warping
2. The output is a HEATMAP of trait values on a template
3. No "visual chaos" - just data on a standardized grid
4. Coverage map shows which regions were observed
    
This is exactly how patternize (Van Belleghem et al. 2018) works
for butterfly wing patterns - we're applying the same methodology
to primate body surfaces.
""")


if __name__ == '__main__':
    main()
