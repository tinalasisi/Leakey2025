#!/usr/bin/env python3
"""
TPS Phenomic Mapping Demo (v2)
==============================
Demonstrates that TPS produces a HEATMAP on a template, not a "warped photo."
This directly addresses Reviewer 2's "visual chaos" concern.

New in v2:
- Convex hull masking to exclude background
- Optional SAM (Segment Anything) integration for better segmentation
- Template visualization panel
- Higher resolution heatmap

Usage:
    # Basic (convex hull masking)
    python tps_demo_v2.py --image_dir OpenApePose/images --ann_file OpenApePose/annotations/oap_all.json

    # With SAM (requires: pip install segment-anything, and model weights)
    python tps_demo_v2.py --image_dir ... --ann_file ... --use_sam --sam_checkpoint sam_vit_h_4b8939.pth

Output:
    tps_demo_figure.png - 6-panel figure for rebuttal
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RBFInterpolator
from scipy.spatial import ConvexHull, Delaunay
from PIL import Image
from pathlib import Path

# Optional SAM import
try:
    from segment_anything import sam_model_registry, SamPredictor
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False

# Landmark definitions from OpenApePose
LANDMARK_NAMES = [
    'nose', 'left_eye', 'right_eye', 'head_top', 'neck',
    'left_shoulder', 'left_elbow', 'left_wrist',
    'right_shoulder', 'right_elbow', 'right_wrist',
    'pelvis', 'left_knee', 'left_ankle', 'right_knee', 'right_ankle'
]

# Skeleton connections for visualization
SKELETON = [(0, 4), (4, 5), (4, 8), (5, 6), (6, 7), (8, 9), (9, 10),
            (4, 11), (11, 12), (12, 13), (11, 14), (14, 15)]

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


def create_body_mask_convex_hull(image_shape, landmarks, visibility):
    """
    Create a body mask using the convex hull of visible landmarks.
    This excludes most background while being simple and fast.
    """
    visible_idx = visibility > 0
    visible_landmarks = landmarks[visible_idx]
    
    # Compute convex hull
    hull = ConvexHull(visible_landmarks)
    hull_points = visible_landmarks[hull.vertices]
    
    # Create mask using Delaunay triangulation for point-in-polygon test
    delaunay = Delaunay(hull_points)
    
    # Create coordinate grid
    h, w = image_shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    points = np.column_stack([xx.ravel(), yy.ravel()])
    
    # Test which points are inside the hull
    inside = delaunay.find_simplex(points) >= 0
    mask = inside.reshape(h, w)
    
    return mask, hull_points


def create_body_mask_sam(image, landmarks, visibility, bbox, sam_predictor):
    """
    Create a body mask using SAM (Segment Anything Model).
    Uses the bounding box and visible landmarks as prompts.
    """
    sam_predictor.set_image(image)
    
    # bbox format: [x, y, width, height] -> [x1, y1, x2, y2]
    x, y, w, h = bbox
    box = np.array([x, y, x + w, y + h])
    
    # Use visible landmarks as point prompts
    visible_idx = visibility > 0
    point_coords = landmarks[visible_idx]
    point_labels = np.ones(len(point_coords))  # All foreground
    
    # Get mask prediction
    masks, scores, logits = sam_predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        box=box,
        multimask_output=True
    )
    
    # Return the mask with highest score
    best_mask = masks[np.argmax(scores)]
    return best_mask


def sample_body_region_with_mask(image, mask, n_samples=3000):
    """Sample pixels only from within the body mask."""
    # Get coordinates of all pixels inside mask
    inside_coords = np.argwhere(mask)  # Returns (y, x) pairs
    
    if len(inside_coords) == 0:
        raise ValueError("Mask is empty - no body pixels found")
    
    # Randomly sample from inside pixels
    np.random.seed(42)
    n_samples = min(n_samples, len(inside_coords))
    sample_indices = np.random.choice(len(inside_coords), n_samples, replace=False)
    sampled_yx = inside_coords[sample_indices]
    
    # Convert to (x, y) format
    sample_points = sampled_yx[:, ::-1].astype(float)
    
    # Get RGB values
    rgb_values = image[sampled_yx[:, 0], sampled_yx[:, 1], :]
    
    return sample_points, rgb_values


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


def create_demo_figure(image, landmarks, visibility, mask, hull_points,
                       template_coords, rgb_values, heatmap, coverage, 
                       output_path, mask_method='convex_hull'):
    """Create the 6-panel demo figure for the rebuttal."""
    
    fig = plt.figure(figsize=(16, 10))
    
    # Panel A: Original image with landmarks and mask overlay
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(image)
    # Show mask as semi-transparent overlay
    mask_overlay = np.zeros((*mask.shape, 4))
    mask_overlay[mask, :] = [0, 1, 0, 0.2]
    ax1.imshow(mask_overlay)
    # Draw landmarks
    visible_idx = visibility > 0
    ax1.scatter(landmarks[visible_idx, 0], landmarks[visible_idx, 1], 
               c='lime', s=100, edgecolors='black', linewidths=2, zorder=5)
    # Draw skeleton
    for i, j in SKELETON:
        if visibility[i] and visibility[j]:
            ax1.plot([landmarks[i, 0], landmarks[j, 0]], 
                    [landmarks[i, 1], landmarks[j, 1]], 'c-', linewidth=2)
    # Draw hull outline
    if hull_points is not None:
        hull_closed = np.vstack([hull_points, hull_points[0]])
        ax1.plot(hull_closed[:, 0], hull_closed[:, 1], 'r-', linewidth=2, label='Body mask')
    ax1.set_title(f'A. Original Image + Body Segmentation\n({mask_method})', 
                 fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # Panel B: Canonical template (empty)
    ax2 = fig.add_subplot(2, 3, 2)
    draw_template(ax2, 'B. Canonical Body Template\n(standardized landmark positions)', 
                  show_labels=True)
    
    # Panel C: Template with mapped sample points
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.set_xlim(-0.1, 1.1)
    ax3.set_ylim(1.1, -0.1)
    for i, j in SKELETON:
        ax3.plot([TEMPLATE_LANDMARKS[i, 0], TEMPLATE_LANDMARKS[j, 0]], 
                [TEMPLATE_LANDMARKS[i, 1], TEMPLATE_LANDMARKS[j, 1]], 
                'gray', linewidth=1, zorder=1)
    colors = rgb_values / 255.0
    ax3.scatter(template_coords[:, 0], template_coords[:, 1], 
               c=colors, s=5, alpha=0.7)
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
    
    2. BODY SEGMENTATION  
       Mask identifies body pixels (green overlay)
    
    3. COORDINATE MAPPING (TPS)
       Each body pixel → template location
       (Panels B→C transformation)
    
    4. TRAIT EXTRACTION
       RGB values sampled at each pixel
    
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
    parser.add_argument('--use_sam', action='store_true',
                        help='Use SAM for segmentation (requires segment-anything package)')
    parser.add_argument('--sam_checkpoint', type=str, default='sam_vit_h_4b8939.pth',
                        help='Path to SAM model checkpoint')
    parser.add_argument('--sam_model_type', type=str, default='vit_h',
                        help='SAM model type (vit_h, vit_l, vit_b)')
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
    
    # Create body mask
    hull_points = None
    if args.use_sam:
        if not SAM_AVAILABLE:
            print("WARNING: SAM not available. Install with: pip install segment-anything")
            print("Falling back to convex hull method.")
            mask, hull_points = create_body_mask_convex_hull(image.shape, landmarks, visibility)
            mask_method = 'convex_hull'
        else:
            print("Using SAM for segmentation...")
            sam = sam_model_registry[args.sam_model_type](checkpoint=args.sam_checkpoint)
            sam_predictor = SamPredictor(sam)
            mask = create_body_mask_sam(image, landmarks, visibility, example['bbox'], sam_predictor)
            mask_method = 'SAM'
    else:
        print("Using convex hull for body mask...")
        mask, hull_points = create_body_mask_convex_hull(image.shape, landmarks, visibility)
        mask_method = 'convex_hull'
    
    print(f"  Body mask: {mask.sum()} pixels ({100*mask.sum()/mask.size:.1f}% of image)")
    
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
    
    # Sample body pixels (only from within mask)
    print("Sampling body region...")
    sample_points, rgb_values = sample_body_region_with_mask(image, mask, n_samples=5000)
    
    # Normalize sample points
    sample_normalized = sample_points.copy()
    sample_normalized[:, 0] /= img_w
    sample_normalized[:, 1] /= img_h
    
    # Map sample points to template coordinates
    print("Mapping samples to template...")
    template_coords = tps_map(sample_normalized)
    
    # Create heatmap
    print("Creating phenomic heatmap...")
    heatmap, coverage, count = create_template_heatmap(template_coords, rgb_values, grid_resolution=50)
    
    # Create figure
    print("Generating figure...")
    create_demo_figure(image, landmarks, visibility, mask, hull_points,
                       template_coords, rgb_values, heatmap, coverage, 
                       args.output, mask_method=mask_method)
    
    print("\n" + "="*70)
    print("DEMO COMPLETE")
    print("="*70)
    print("""
This figure demonstrates that:
    
1. Body segmentation isolates the subject from background
2. TPS is a COORDINATE MAPPING, not image warping
3. The output is a HEATMAP of trait values on a standardized template
4. No "visual chaos" - just data on a canonical grid
5. Coverage map shows which regions were observed
    
This is exactly how patternize (Van Belleghem et al. 2018) works
for butterfly wing patterns - we're applying the same methodology
to primate body surfaces.

To use SAM for better segmentation:
    pip install segment-anything
    # Download model weights (~2.5GB):
    wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
    # Then run with:
    python tps_demo_v2.py --use_sam --sam_checkpoint sam_vit_h_4b8939.pth ...
""")


if __name__ == '__main__':
    main()
