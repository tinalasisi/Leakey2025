#!/usr/bin/env python3
"""
TPS Phenomic Mapping Demo (v3)
==============================
Samples pixels along skeleton lines with a radius around each point.
Includes background color filtering to exclude sky/foliage.

Usage:
    python tps_demo_v3.py --image_dir OpenApePose/images --ann_file OpenApePose/annotations/oap_all.json

Output:
    tps_demo_figure.png - 6-panel figure for rebuttal
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
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

# Skeleton connections for visualization
SKELETON = [(0, 4), (4, 5), (4, 8), (5, 6), (6, 7), (8, 9), (9, 10),
            (4, 11), (11, 12), (12, 13), (11, 14), (14, 15),
            # Add head connections
            (0, 1), (0, 2), (0, 3), (1, 3), (2, 3)]

# Canonical template landmarks
TEMPLATE_LANDMARKS = np.array([
    [0.50, 0.08],  # 0: nose
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
    [0.50, 0.50],  # 11: pelvis
    [0.38, 0.70],  # 12: left_knee
    [0.35, 0.92],  # 13: left_ankle
    [0.62, 0.70],  # 14: right_knee
    [0.65, 0.92],  # 15: right_ankle
])


def load_annotations(ann_file):
    """Load OpenApePose annotations."""
    with open(ann_file) as f:
        data = json.load(f)
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
    """Compute TPS interpolation from source to target coordinates."""
    rbf_x = RBFInterpolator(source_points, target_points[:, 0], kernel='thin_plate_spline')
    rbf_y = RBFInterpolator(source_points, target_points[:, 1], kernel='thin_plate_spline')
    
    def mapping_function(points):
        x_mapped = rbf_x(points)
        y_mapped = rbf_y(points)
        return np.column_stack([x_mapped, y_mapped])
    
    return mapping_function


def is_background_color(rgb, bg_thresholds=None):
    """
    Detect if a pixel is likely background (sky, foliage, etc.)
    
    Returns True if pixel should be EXCLUDED.
    """
    r, g, b = rgb[0], rgb[1], rgb[2]
    
    # Sky detection: high blue, blue > red, blue > green
    if b > 150 and b > r and b > g * 0.9:
        return True
    
    # Bright sky: very bright and bluish
    if b > 180 and r > 150 and g > 150 and b >= r and b >= g:
        return True
    
    # Green foliage: high green relative to others
    if g > 100 and g > r * 1.2 and g > b * 1.1:
        return True
    
    # Very bright (overexposed sky)
    if r > 240 and g > 240 and b > 240:
        return True
    
    return False


def sample_along_skeleton(image, landmarks, visibility, radius=10, points_per_segment=20):
    """
    Sample pixels along skeleton lines with a radius around each point.
    
    Args:
        image: RGB image array
        landmarks: (16, 2) array of landmark coordinates
        visibility: (16,) array of visibility flags
        radius: pixel radius around skeleton to sample
        points_per_segment: number of points to sample along each skeleton segment
    
    Returns:
        sample_points: (N, 2) array of sampled pixel coordinates
        rgb_values: (N, 3) array of RGB values
        kept_mask: boolean array indicating which samples passed background filter
    """
    h, w = image.shape[:2]
    all_points = []
    all_rgb = []
    
    # Sample along each skeleton segment
    for i, j in SKELETON:
        if not (visibility[i] and visibility[j]):
            continue
        
        p1 = landmarks[i]
        p2 = landmarks[j]
        
        # Generate points along the line
        for t in np.linspace(0, 1, points_per_segment):
            center = p1 + t * (p2 - p1)
            
            # Sample in a circle around this point
            for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
                for r in np.linspace(0, radius, 3):
                    px = int(center[0] + r * np.cos(angle))
                    py = int(center[1] + r * np.sin(angle))
                    
                    # Bounds check
                    if 0 <= px < w and 0 <= py < h:
                        all_points.append([px, py])
                        all_rgb.append(image[py, px, :])
    
    # Also sample around each landmark directly
    for i in range(len(landmarks)):
        if not visibility[i]:
            continue
        cx, cy = landmarks[i]
        for angle in np.linspace(0, 2*np.pi, 16, endpoint=False):
            for r in np.linspace(0, radius * 1.5, 5):
                px = int(cx + r * np.cos(angle))
                py = int(cy + r * np.sin(angle))
                if 0 <= px < w and 0 <= py < h:
                    all_points.append([px, py])
                    all_rgb.append(image[py, px, :])
    
    sample_points = np.array(all_points, dtype=float)
    rgb_values = np.array(all_rgb)
    
    # Filter out background pixels
    kept_mask = np.array([not is_background_color(rgb) for rgb in rgb_values])
    
    return sample_points, rgb_values, kept_mask


def create_template_heatmap(template_coords, rgb_values, grid_resolution=50):
    """Create a heatmap on the template by aggregating RGB values."""
    color_sum = np.zeros((grid_resolution, grid_resolution, 3))
    count = np.zeros((grid_resolution, grid_resolution))
    
    for (tx, ty), rgb in zip(template_coords, rgb_values):
        if 0 <= tx <= 1 and 0 <= ty <= 1:
            gx = int(tx * (grid_resolution - 1))
            gy = int(ty * (grid_resolution - 1))
            color_sum[gy, gx, :] += rgb
            count[gy, gx] += 1
    
    with np.errstate(divide='ignore', invalid='ignore'):
        avg_color = color_sum / count[:, :, np.newaxis]
        avg_color = np.nan_to_num(avg_color, nan=255)
    
    coverage = count > 0
    return avg_color.astype(np.uint8), coverage, count


def draw_template(ax, title, show_labels=False):
    """Draw the canonical template skeleton on an axis."""
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(1.1, -0.1)
    
    for i, j in SKELETON:
        ax.plot([TEMPLATE_LANDMARKS[i, 0], TEMPLATE_LANDMARKS[j, 0]], 
               [TEMPLATE_LANDMARKS[i, 1], TEMPLATE_LANDMARKS[j, 1]], 
               'k-', linewidth=2, zorder=1)
    
    ax.scatter(TEMPLATE_LANDMARKS[:, 0], TEMPLATE_LANDMARKS[:, 1], 
               c='black', s=150, marker='o', zorder=5, edgecolors='white', linewidths=2)
    
    if show_labels:
        for i, name in enumerate(LANDMARK_NAMES):
            ax.annotate(f'{i}', TEMPLATE_LANDMARKS[i], fontsize=8, ha='center', va='bottom',
                       xytext=(0, 5), textcoords='offset points')
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_aspect('equal')
    ax.set_xlabel('Template X')
    ax.set_ylabel('Template Y')


def create_demo_figure(image, landmarks, visibility, 
                       sample_points, rgb_values, kept_mask,
                       template_coords, heatmap, coverage, 
                       output_path, radius):
    """Create the 6-panel demo figure for the rebuttal."""
    
    fig = plt.figure(figsize=(16, 10))
    
    # Panel A: Original image with landmarks and sampled regions
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(image)
    # Show kept samples as green dots, filtered as red
    ax1.scatter(sample_points[kept_mask, 0], sample_points[kept_mask, 1], 
               c='lime', s=1, alpha=0.3, label='Body pixels')
    ax1.scatter(sample_points[~kept_mask, 0], sample_points[~kept_mask, 1], 
               c='red', s=1, alpha=0.3, label='Background (filtered)')
    # Draw landmarks
    visible_idx = visibility > 0
    ax1.scatter(landmarks[visible_idx, 0], landmarks[visible_idx, 1], 
               c='yellow', s=100, edgecolors='black', linewidths=2, zorder=5)
    # Draw skeleton
    for i, j in SKELETON:
        if visibility[i] and visibility[j]:
            ax1.plot([landmarks[i, 0], landmarks[j, 0]], 
                    [landmarks[i, 1], landmarks[j, 1]], 'c-', linewidth=2)
    ax1.legend(loc='upper right', fontsize=8)
    ax1.set_title(f'A. Skeleton Sampling (radius={radius}px)\nGreen=kept, Red=background filtered', 
                 fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # Panel B: Canonical template (empty)
    ax2 = fig.add_subplot(2, 3, 2)
    draw_template(ax2, 'B. Canonical Body Template\n(standardized landmark positions)', 
                  show_labels=True)
    
    # Panel C: Template with mapped sample points (only kept ones)
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.set_xlim(-0.1, 1.1)
    ax3.set_ylim(1.1, -0.1)
    for i, j in SKELETON:
        ax3.plot([TEMPLATE_LANDMARKS[i, 0], TEMPLATE_LANDMARKS[j, 0]], 
                [TEMPLATE_LANDMARKS[i, 1], TEMPLATE_LANDMARKS[j, 1]], 
                'gray', linewidth=1, zorder=1)
    # Only show kept samples
    kept_template = template_coords[kept_mask]
    kept_rgb = rgb_values[kept_mask]
    colors = kept_rgb / 255.0
    ax3.scatter(kept_template[:, 0], kept_template[:, 1], 
               c=colors, s=8, alpha=0.8)
    ax3.set_title('C. Body Pixels Mapped to Template\n(TPS coordinate transformation)', 
                 fontsize=12, fontweight='bold')
    ax3.set_aspect('equal')
    ax3.set_xlabel('Template X')
    ax3.set_ylabel('Template Y')
    
    # Panel D: Color heatmap on template (THE KEY OUTPUT)
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.imshow(heatmap, extent=[0, 1, 1, 0])
    for i, j in SKELETON:
        ax4.plot([TEMPLATE_LANDMARKS[i, 0], TEMPLATE_LANDMARKS[j, 0]], 
                [TEMPLATE_LANDMARKS[i, 1], TEMPLATE_LANDMARKS[j, 1]], 
                'white', linewidth=1, alpha=0.5)
    ax4.scatter(TEMPLATE_LANDMARKS[:, 0], TEMPLATE_LANDMARKS[:, 1], 
               c='white', s=50, marker='o', edgecolors='black', linewidths=1, zorder=5)
    ax4.set_title('D. PHENOMIC MAP: Color at Template Locations\n(aggregated data, NOT a warped photo)', 
                 fontsize=12, fontweight='bold')
    ax4.set_xlabel('Template X')
    ax4.set_ylabel('Template Y')
    
    # Panel E: Coverage map
    ax5 = fig.add_subplot(2, 3, 5)
    im = ax5.imshow(coverage.astype(float), extent=[0, 1, 1, 0], cmap='Greens', vmin=0, vmax=1)
    for i, j in SKELETON:
        ax5.plot([TEMPLATE_LANDMARKS[i, 0], TEMPLATE_LANDMARKS[j, 0]], 
                [TEMPLATE_LANDMARKS[i, 1], TEMPLATE_LANDMARKS[j, 1]], 
                'black', linewidth=1, alpha=0.5)
    ax5.scatter(TEMPLATE_LANDMARKS[:, 0], TEMPLATE_LANDMARKS[:, 1], 
               c='black', s=50, marker='o', edgecolors='white', linewidths=1, zorder=5)
    ax5.set_title('E. Coverage Map\n(which template regions were observed)', 
                 fontsize=12, fontweight='bold')
    ax5.set_xlabel('Template X')
    ax5.set_ylabel('Template Y')
    plt.colorbar(im, ax=ax5, label='Observed', shrink=0.8)
    
    # Panel F: Text explanation
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    explanation = """
    Pipeline Summary:
    
    1. POSE DETECTION
       Landmarks detected in image (Panel A)
    
    2. SKELETON SAMPLING
       Pixels sampled along skeleton with radius
       Background pixels filtered by color
    
    3. COORDINATE MAPPING (TPS)
       Each body pixel → template location
       (Panels B→C transformation)
    
    4. TRAIT EXTRACTION
       RGB values at each sampled pixel
    
    5. PHENOMIC MAP
       Values aggregated at template locations
       (Panel D: THE OUTPUT)
    
    KEY POINT: No "warped image" is created.
    TPS provides coordinate lookup, not image
    deformation. Output is a data visualization.
    """
    ax6.text(0.1, 0.95, explanation, transform=ax6.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
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
    parser.add_argument('--radius', type=int, default=15,
                        help='Pixel radius around skeleton to sample (default: 15)')
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
    
    # Sample along skeleton with radius
    print(f"Sampling along skeleton (radius={args.radius}px)...")
    sample_points, rgb_values, kept_mask = sample_along_skeleton(
        image, landmarks, visibility, radius=args.radius, points_per_segment=25
    )
    print(f"  Total samples: {len(sample_points)}")
    print(f"  Kept (body): {kept_mask.sum()} ({100*kept_mask.mean():.1f}%)")
    print(f"  Filtered (background): {(~kept_mask).sum()} ({100*(~kept_mask).mean():.1f}%)")
    
    # Normalize landmarks for TPS
    visible_idx = visibility > 0
    source_landmarks = landmarks[visible_idx]
    target_landmarks = TEMPLATE_LANDMARKS[visible_idx]
    
    img_h, img_w = image.shape[:2]
    source_normalized = source_landmarks.astype(float)
    source_normalized[:, 0] /= img_w
    source_normalized[:, 1] /= img_h
    
    # Compute TPS mapping
    print("Computing TPS mapping...")
    tps_map = compute_tps_mapping(source_normalized, target_landmarks)
    
    # Normalize sample points
    sample_normalized = sample_points.copy()
    sample_normalized[:, 0] /= img_w
    sample_normalized[:, 1] /= img_h
    
    # Map ALL sample points to template coordinates (for visualization)
    print("Mapping samples to template...")
    template_coords = tps_map(sample_normalized)
    
    # Create heatmap using ONLY kept samples
    print("Creating phenomic heatmap...")
    heatmap, coverage, count = create_template_heatmap(
        template_coords[kept_mask], 
        rgb_values[kept_mask], 
        grid_resolution=50
    )
    
    # Create figure
    print("Generating figure...")
    create_demo_figure(image, landmarks, visibility,
                       sample_points, rgb_values, kept_mask,
                       template_coords, heatmap, coverage, 
                       args.output, args.radius)
    
    print("\n" + "="*70)
    print("DEMO COMPLETE")
    print("="*70)
    print(f"""
This figure demonstrates that:
    
1. Sampling along skeleton captures body pixels, not whole image
2. Background filtering removes sky/foliage pixels
3. TPS is a COORDINATE MAPPING, not image warping
4. The output is a HEATMAP of trait values on a template
5. Coverage map shows which regions were observed
    
Try different radius values with --radius flag (default: 15)
Larger radius = more samples but may catch more background
""")


if __name__ == '__main__':
    main()
