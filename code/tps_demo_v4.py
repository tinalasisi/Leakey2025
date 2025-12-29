#!/usr/bin/env python3
"""
TPS Phenomic Mapping Demo (v4)
==============================
Samples pixels along skeleton lines and maps them using LINEAR INTERPOLATION
along each segment (not global TPS). This handles bent limbs correctly.

Also includes a body outline template for clearer visualization.

Usage:
    python tps_demo_v4.py --image_dir OpenApePose/images --ann_file OpenApePose/annotations/oap_all.json

Output:
    tps_demo_figure.png - 6-panel figure for rebuttal
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

# Landmark definitions from OpenApePose
LANDMARK_NAMES = [
    'nose', 'left_eye', 'right_eye', 'head_top', 'neck',
    'left_shoulder', 'left_elbow', 'left_wrist',
    'right_shoulder', 'right_elbow', 'right_wrist',
    'pelvis', 'left_knee', 'left_ankle', 'right_knee', 'right_ankle'
]

# Skeleton connections for sampling
SKELETON = [
    (0, 4),   # nose to neck
    (4, 5),   # neck to left shoulder
    (4, 8),   # neck to right shoulder
    (5, 6),   # left shoulder to elbow
    (6, 7),   # left elbow to wrist
    (8, 9),   # right shoulder to elbow
    (9, 10),  # right elbow to wrist
    (4, 11),  # neck to pelvis
    (11, 12), # pelvis to left knee
    (12, 13), # left knee to ankle
    (11, 14), # pelvis to right knee
    (14, 15), # right knee to ankle
    # Head
    (0, 1), (0, 2), (0, 3), (1, 3), (2, 3),
    (5, 8),  # shoulder to shoulder (across chest)
]

# Canonical template landmarks (normalized 0-1)
TEMPLATE_LANDMARKS = np.array([
    [0.50, 0.08],  # 0: nose
    [0.45, 0.05],  # 1: left_eye
    [0.55, 0.05],  # 2: right_eye
    [0.50, 0.02],  # 3: head_top
    [0.50, 0.18],  # 4: neck
    [0.35, 0.22],  # 5: left_shoulder
    [0.22, 0.38],  # 6: left_elbow
    [0.15, 0.52],  # 7: left_wrist
    [0.65, 0.22],  # 8: right_shoulder
    [0.78, 0.38],  # 9: right_elbow
    [0.85, 0.52],  # 10: right_wrist
    [0.50, 0.52],  # 11: pelvis
    [0.40, 0.72],  # 12: left_knee
    [0.38, 0.95],  # 13: left_ankle
    [0.60, 0.72],  # 14: right_knee
    [0.62, 0.95],  # 15: right_ankle
])


def load_annotations(ann_file):
    with open(ann_file) as f:
        data = json.load(f)
    if isinstance(data, dict) and 'data' in data:
        return data['data']
    return data


def find_good_example(annotations, species_filter=None, min_visible=14):
    for ann in annotations:
        if species_filter and ann['species'].lower() != species_filter.lower():
            continue
        if sum(ann['visibility']) >= min_visible:
            return ann
    return None


def parse_landmarks(ann):
    landmarks = np.array(ann['landmarks']).reshape(-1, 2)
    visibility = np.array(ann['visibility'])
    return landmarks, visibility


def is_background_color(rgb):
    """Detect if a pixel is likely background."""
    r, g, b = rgb[0], rgb[1], rgb[2]
    
    # Sky: high blue
    if b > 150 and b > r and b > g * 0.9:
        return True
    # Bright sky
    if b > 180 and r > 150 and g > 150 and b >= r and b >= g:
        return True
    # Green foliage
    if g > 100 and g > r * 1.2 and g > b * 1.1:
        return True
    # Overexposed
    if r > 240 and g > 240 and b > 240:
        return True
    return False


def point_to_line_projection(point, line_start, line_end):
    """
    Project a point onto a line segment.
    Returns:
        t: parameter along line (0 = start, 1 = end)
        perp_dist: signed perpendicular distance from line
        perp_vec: unit vector perpendicular to line
    """
    line_vec = line_end - line_start
    line_len = np.linalg.norm(line_vec)
    if line_len < 1e-6:
        return 0.5, 0, np.array([0, 1])
    
    line_unit = line_vec / line_len
    point_vec = point - line_start
    
    # Parameter t along line
    t = np.dot(point_vec, line_unit) / line_len
    
    # Perpendicular vector (rotate line_unit 90 degrees)
    perp_unit = np.array([-line_unit[1], line_unit[0]])
    
    # Signed perpendicular distance
    perp_dist = np.dot(point_vec, perp_unit)
    
    return t, perp_dist, perp_unit


def map_point_along_segment(t, perp_dist, template_start, template_end, scale_perp=True):
    """
    Map a point from image segment to template segment.
    
    Args:
        t: parameter along segment (0-1)
        perp_dist: perpendicular distance from line in image pixels
        template_start, template_end: template segment endpoints
        scale_perp: if True, scale perpendicular distance by segment length ratio
    """
    template_vec = template_end - template_start
    template_len = np.linalg.norm(template_vec)
    
    if template_len < 1e-6:
        return template_start
    
    template_unit = template_vec / template_len
    template_perp = np.array([-template_unit[1], template_unit[0]])
    
    # Position along template segment
    pos = template_start + t * template_vec
    
    # Add perpendicular offset (scaled to template coordinates)
    # The perp_dist is in image pixels, we need to scale it
    # Use a fixed scaling factor based on typical body width
    perp_scale = 0.001  # Adjust this to control "thickness" mapping
    pos = pos + perp_dist * perp_scale * template_perp
    
    return pos


def sample_along_skeleton_with_mapping(image, landmarks, visibility, template_landmarks,
                                        radius=15, points_per_segment=30):
    """
    Sample pixels along skeleton and map directly to template using linear interpolation.
    
    Returns:
        sample_points: image coordinates of samples
        template_coords: corresponding template coordinates  
        rgb_values: RGB values at each sample
        kept_mask: which samples passed background filter
        segment_ids: which skeleton segment each sample belongs to
    """
    h, w = image.shape[:2]
    
    all_image_points = []
    all_template_points = []
    all_rgb = []
    all_segment_ids = []
    
    for seg_idx, (i, j) in enumerate(SKELETON):
        if not (visibility[i] and visibility[j]):
            continue
        
        # Image segment
        img_start = landmarks[i]
        img_end = landmarks[j]
        img_vec = img_end - img_start
        img_len = np.linalg.norm(img_vec)
        
        # Template segment
        tmpl_start = template_landmarks[i]
        tmpl_end = template_landmarks[j]
        
        if img_len < 1:
            continue
        
        # Sample along the segment
        for t in np.linspace(0, 1, points_per_segment):
            center = img_start + t * img_vec
            
            # Sample in a radius around this point
            for angle in np.linspace(0, 2*np.pi, 12, endpoint=False):
                for r in [0, radius*0.33, radius*0.67, radius]:
                    px = center[0] + r * np.cos(angle)
                    py = center[1] + r * np.sin(angle)
                    
                    pxi, pyi = int(px), int(py)
                    if not (0 <= pxi < w and 0 <= pyi < h):
                        continue
                    
                    # Get projection onto image segment
                    point = np.array([px, py])
                    t_proj, perp_dist, _ = point_to_line_projection(point, img_start, img_end)
                    
                    # Map to template
                    template_point = map_point_along_segment(
                        t_proj, perp_dist, tmpl_start, tmpl_end
                    )
                    
                    all_image_points.append([px, py])
                    all_template_points.append(template_point)
                    all_rgb.append(image[pyi, pxi, :])
                    all_segment_ids.append(seg_idx)
    
    sample_points = np.array(all_image_points)
    template_coords = np.array(all_template_points)
    rgb_values = np.array(all_rgb)
    segment_ids = np.array(all_segment_ids)
    
    # Filter background
    kept_mask = np.array([not is_background_color(rgb) for rgb in rgb_values])
    
    return sample_points, template_coords, rgb_values, kept_mask, segment_ids


def create_body_mask_on_grid(template_landmarks, grid_resolution=60):
    """
    Create a binary mask of the body region on the template grid.
    Returns a boolean array where True = inside body, False = background.
    """
    from matplotlib.path import Path as MplPath
    
    mask = np.zeros((grid_resolution, grid_resolution), dtype=bool)
    
    # Create coordinate grid
    x = np.linspace(0, 1, grid_resolution)
    y = np.linspace(0, 1, grid_resolution)
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack([xx.ravel(), yy.ravel()])
    
    # Head circle
    head_center = (template_landmarks[0] + template_landmarks[3]) / 2
    head_radius = 0.07
    head_dist = np.sqrt((points[:, 0] - head_center[0])**2 + (points[:, 1] - head_center[1])**2)
    head_mask = head_dist < head_radius
    
    # Torso polygon
    torso_verts = [
        template_landmarks[5],   # left shoulder
        template_landmarks[8],   # right shoulder
        template_landmarks[11] + [0.10, 0.02],  # right hip
        template_landmarks[11] - [0.10, -0.02],  # left hip
    ]
    torso_path = MplPath(torso_verts)
    torso_mask = torso_path.contains_points(points)
    
    # Limbs as thick lines (rectangles)
    def limb_mask(p1, p2, width=0.06):
        """Create mask for a limb segment."""
        vec = p2 - p1
        length = np.linalg.norm(vec)
        if length < 0.01:
            return np.zeros(len(points), dtype=bool)
        unit = vec / length
        perp = np.array([-unit[1], unit[0]])
        
        # Rectangle corners
        corners = [
            p1 - perp * width/2,
            p1 + perp * width/2,
            p2 + perp * width/2,
            p2 - perp * width/2,
        ]
        path = MplPath(corners)
        return path.contains_points(points)
    
    # Left arm
    left_arm = (limb_mask(template_landmarks[5], template_landmarks[6], 0.06) |
                limb_mask(template_landmarks[6], template_landmarks[7], 0.05))
    
    # Right arm
    right_arm = (limb_mask(template_landmarks[8], template_landmarks[9], 0.06) |
                 limb_mask(template_landmarks[9], template_landmarks[10], 0.05))
    
    # Left leg
    left_leg = (limb_mask(template_landmarks[11] - [0.05, 0], template_landmarks[12], 0.07) |
                limb_mask(template_landmarks[12], template_landmarks[13], 0.06))
    
    # Right leg
    right_leg = (limb_mask(template_landmarks[11] + [0.05, 0], template_landmarks[14], 0.07) |
                 limb_mask(template_landmarks[14], template_landmarks[15], 0.06))
    
    # Combine all body parts
    body_mask = head_mask | torso_mask | left_arm | right_arm | left_leg | right_leg
    
    return body_mask.reshape(grid_resolution, grid_resolution)


def create_template_heatmap(template_coords, rgb_values, grid_resolution=60, body_mask=None):
    """Create heatmap by aggregating RGB values on template grid."""
    color_sum = np.zeros((grid_resolution, grid_resolution, 3))
    count = np.zeros((grid_resolution, grid_resolution))
    
    for (tx, ty), rgb in zip(template_coords, rgb_values):
        if 0 <= tx <= 1 and 0 <= ty <= 1:
            gx = int(tx * (grid_resolution - 1))
            gy = int(ty * (grid_resolution - 1))
            color_sum[gy, gx, :] += rgb
            count[gy, gx] += 1
    
    # Create the color heatmap
    with np.errstate(divide='ignore', invalid='ignore'):
        avg_color = color_sum / count[:, :, np.newaxis]
    
    # Where we have no data, use light gray (will be masked by body outline)
    avg_color = np.nan_to_num(avg_color, nan=220)
    
    # If we have a body mask, set background to darker gray
    if body_mask is not None:
        # Background (outside body) = medium gray
        avg_color[~body_mask] = [180, 180, 180]
    
    coverage = count > 0
    return avg_color.astype(np.uint8), coverage, count


def draw_template_with_outline(ax, template_landmarks, title, show_labels=False):
    """Draw template with body outline and skeleton."""
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(1.05, -0.05)
    
    # Draw gray background with white body shape
    body_mask = create_body_mask_on_grid(template_landmarks, grid_resolution=100)
    body_bg = np.ones((100, 100, 3)) * 0.7  # Gray background
    body_bg[body_mask] = [0.95, 0.95, 0.95]  # Light gray/white body
    ax.imshow(body_bg, extent=[0, 1, 1, 0], zorder=0)
    
    # Draw skeleton
    for i, j in SKELETON:
        ax.plot([template_landmarks[i, 0], template_landmarks[j, 0]], 
               [template_landmarks[i, 1], template_landmarks[j, 1]], 
               'k-', linewidth=2, zorder=2)
    
    # Draw landmarks
    ax.scatter(template_landmarks[:, 0], template_landmarks[:, 1], 
               c='black', s=80, marker='o', zorder=5, edgecolors='white', linewidths=1.5)
    
    if show_labels:
        offsets = {
            0: (0, 8), 1: (-8, 0), 2: (8, 0), 3: (0, -8), 4: (8, 0),
            5: (-10, 0), 6: (-10, 0), 7: (-10, 0),
            8: (10, 0), 9: (10, 0), 10: (10, 0),
            11: (10, 0), 12: (-10, 0), 13: (-10, 0), 14: (10, 0), 15: (10, 0)
        }
        for i in range(len(template_landmarks)):
            off = offsets.get(i, (5, 0))
            ax.annotate(f'{i}', template_landmarks[i], fontsize=7, ha='center', va='center',
                       xytext=off, textcoords='offset points')
    
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_aspect('equal')
    ax.set_xlabel('Template X')
    ax.set_ylabel('Template Y')


def create_demo_figure(image, landmarks, visibility, 
                       sample_points, template_coords, rgb_values, kept_mask,
                       heatmap, coverage, output_path, radius):
    """Create the 6-panel demo figure."""
    
    fig = plt.figure(figsize=(16, 10))
    
    # Panel A: Original image with sampling
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(image)
    ax1.scatter(sample_points[kept_mask, 0], sample_points[kept_mask, 1], 
               c='lime', s=1, alpha=0.4, label='Body pixels')
    ax1.scatter(sample_points[~kept_mask, 0], sample_points[~kept_mask, 1], 
               c='red', s=1, alpha=0.3, label='Background')
    visible_idx = visibility > 0
    ax1.scatter(landmarks[visible_idx, 0], landmarks[visible_idx, 1], 
               c='yellow', s=80, edgecolors='black', linewidths=2, zorder=5)
    for i, j in SKELETON:
        if visibility[i] and visibility[j]:
            ax1.plot([landmarks[i, 0], landmarks[j, 0]], 
                    [landmarks[i, 1], landmarks[j, 1]], 'c-', linewidth=2)
    ax1.legend(loc='upper right', fontsize=8, markerscale=5)
    ax1.set_title(f'A. Skeleton Sampling (radius={radius}px)', fontsize=11, fontweight='bold')
    ax1.axis('off')
    
    # Panel B: Template with body outline
    ax2 = fig.add_subplot(2, 3, 2)
    draw_template_with_outline(ax2, TEMPLATE_LANDMARKS, 
                               'B. Canonical Body Template', show_labels=True)
    
    # Panel C: Mapped samples on template with body outline
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.set_xlim(-0.05, 1.05)
    ax3.set_ylim(1.05, -0.05)
    
    # Draw gray background with body shape in white
    body_mask = create_body_mask_on_grid(TEMPLATE_LANDMARKS, grid_resolution=100)
    body_bg = np.ones((100, 100, 3)) * 0.7  # Gray
    body_bg[body_mask] = [1.0, 1.0, 1.0]  # White body
    ax3.imshow(body_bg, extent=[0, 1, 1, 0], zorder=0)
    
    # Draw skeleton
    for i, j in SKELETON:
        ax3.plot([TEMPLATE_LANDMARKS[i, 0], TEMPLATE_LANDMARKS[j, 0]], 
                [TEMPLATE_LANDMARKS[i, 1], TEMPLATE_LANDMARKS[j, 1]], 
                'gray', linewidth=1, zorder=1)
    
    # Plot samples
    kept_template = template_coords[kept_mask]
    kept_rgb = rgb_values[kept_mask]
    colors = kept_rgb / 255.0
    ax3.scatter(kept_template[:, 0], kept_template[:, 1], c=colors, s=6, alpha=0.8, zorder=2)
    ax3.set_title('C. Pixels Mapped to Template\n(linear interpolation along segments)', 
                 fontsize=11, fontweight='bold')
    ax3.set_aspect('equal')
    ax3.set_xlabel('Template X')
    ax3.set_ylabel('Template Y')
    
    # Panel D: Phenomic heatmap with body mask
    ax4 = fig.add_subplot(2, 3, 4)
    
    # Apply body mask to heatmap - set background to gray
    body_mask = create_body_mask_on_grid(TEMPLATE_LANDMARKS, grid_resolution=heatmap.shape[0])
    heatmap_display = heatmap.copy()
    heatmap_display[~body_mask] = [180, 180, 180]  # Gray background
    
    ax4.imshow(heatmap_display, extent=[0, 1, 1, 0])
    for i, j in SKELETON:
        ax4.plot([TEMPLATE_LANDMARKS[i, 0], TEMPLATE_LANDMARKS[j, 0]], 
                [TEMPLATE_LANDMARKS[i, 1], TEMPLATE_LANDMARKS[j, 1]], 
                'white', linewidth=1, alpha=0.5)
    ax4.scatter(TEMPLATE_LANDMARKS[:, 0], TEMPLATE_LANDMARKS[:, 1], 
               c='white', s=40, marker='o', edgecolors='black', linewidths=1, zorder=5)
    ax4.set_title('D. PHENOMIC MAP\n(aggregated color data on template)', 
                 fontsize=11, fontweight='bold')
    ax4.set_xlabel('Template X')
    ax4.set_ylabel('Template Y')
    
    # Panel E: Coverage with body mask
    ax5 = fig.add_subplot(2, 3, 5)
    
    # Create a colored coverage image:
    # - Gray background (outside body)
    # - Red/pink = body region, no data
    # - Green = body region with data
    coverage_rgb = np.ones((coverage.shape[0], coverage.shape[1], 3)) * 0.7  # Gray background
    
    # Get body mask
    body_mask = create_body_mask_on_grid(TEMPLATE_LANDMARKS, grid_resolution=coverage.shape[0])
    
    # Body without data = light red/pink
    missing_body = body_mask & ~coverage
    coverage_rgb[missing_body] = [1.0, 0.7, 0.7]  # Light red
    
    # Body with data = green
    has_data = body_mask & coverage
    coverage_rgb[has_data] = [0.3, 0.8, 0.3]  # Green
    
    ax5.imshow(coverage_rgb, extent=[0, 1, 1, 0])
    
    # Draw skeleton overlay
    for i, j in SKELETON:
        ax5.plot([TEMPLATE_LANDMARKS[i, 0], TEMPLATE_LANDMARKS[j, 0]], 
                [TEMPLATE_LANDMARKS[i, 1], TEMPLATE_LANDMARKS[j, 1]], 
                'black', linewidth=1, alpha=0.5)
    ax5.scatter(TEMPLATE_LANDMARKS[:, 0], TEMPLATE_LANDMARKS[:, 1], 
               c='black', s=40, marker='o', edgecolors='white', linewidths=1, zorder=5)
    ax5.set_title('E. Coverage Map\nGreen=observed, Pink=missing, Gray=background', 
                 fontsize=11, fontweight='bold')
    ax5.set_xlabel('Template X')
    ax5.set_ylabel('Template Y')
    
    # Panel F: Explanation
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    explanation = """
    Pipeline Summary:
    
    1. POSE DETECTION
       16 landmarks detected (Panel A)
    
    2. SKELETON SAMPLING
       Pixels sampled along skeleton segments
       Background filtered by color
    
    3. COORDINATE MAPPING
       Linear interpolation along each segment:
       - Position along bent arm → same position
         along straight template arm
       - NO global warping artifacts
    
    4. PHENOMIC MAP (Panel D)
       RGB values aggregated at template locations
    
    ─────────────────────────────────
    KEY: This is DATA on a template,
    not a "warped photograph"
    ─────────────────────────────────
    
    Same approach as patternize
    (Van Belleghem et al. 2018)
    for butterfly wing patterns.
    """
    ax6.text(0.05, 0.95, explanation, transform=ax6.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='TPS Phenomic Mapping Demo v4')
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--ann_file', type=str, required=True)
    parser.add_argument('--species', type=str, default='Siamang')
    parser.add_argument('--output', type=str, default='tps_demo_figure.png')
    parser.add_argument('--radius', type=int, default=15)
    args = parser.parse_args()
    
    print("Loading annotations...")
    annotations = load_annotations(args.ann_file)
    print(f"Loaded {len(annotations)} annotations")
    
    print(f"Finding {args.species} with good visibility...")
    example = find_good_example(annotations, species_filter=args.species, min_visible=15)
    if example is None:
        example = find_good_example(annotations, min_visible=15)
    
    if example is None:
        print("ERROR: No suitable image found!")
        return
    
    print(f"Using: {example['file']} ({example['species']})")
    print(f"  Visible landmarks: {sum(example['visibility'])}/16")
    
    image_path = Path(args.image_dir) / example['file']
    image = np.array(Image.open(image_path))
    print(f"  Image size: {image.shape}")
    
    landmarks, visibility = parse_landmarks(example)
    
    print(f"Sampling along skeleton (radius={args.radius}px)...")
    sample_points, template_coords, rgb_values, kept_mask, _ = \
        sample_along_skeleton_with_mapping(image, landmarks, visibility, 
                                           TEMPLATE_LANDMARKS, radius=args.radius)
    
    print(f"  Total samples: {len(sample_points)}")
    print(f"  Kept: {kept_mask.sum()} ({100*kept_mask.mean():.1f}%)")
    print(f"  Filtered: {(~kept_mask).sum()}")
    
    print("Creating heatmap...")
    heatmap, coverage, _ = create_template_heatmap(
        template_coords[kept_mask], rgb_values[kept_mask], grid_resolution=60
    )
    
    print("Generating figure...")
    create_demo_figure(image, landmarks, visibility,
                       sample_points, template_coords, rgb_values, kept_mask,
                       heatmap, coverage, args.output, args.radius)
    
    print("\n" + "="*60)
    print("DEMO COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()
