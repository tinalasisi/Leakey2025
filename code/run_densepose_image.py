#!/usr/bin/env python3
"""
DensePose inference on a single image using detectron2 directly.

This bypasses Zamba (which only supports video) and uses detectron2-densepose
directly to get per-pixel IUV predictions.

Usage:
    python code/run_densepose_image.py data/chimp_test.jpg

Requires GPU for reasonable speed, but can run on CPU (slowly).

Author: Tina Lasisi
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import cv2

def run_densepose(image_path, output_dir, use_cpu=False):
    """
    Run DensePose on a single image and extract IUV predictions.
    """
    import torch
    import cv2
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    
    # Import DensePose components
    from densepose import add_densepose_config
    from densepose.vis.extractor import DensePoseResultExtractor
    from densepose.vis.densepose_results import DensePoseResultsFineSegmentationVisualizer
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Setup config
    cfg = get_cfg()
    add_densepose_config(cfg)
    
    # Use the standard DensePose model (trained on humans, but let's see what happens with chimps)
    # For chimps specifically, we'd want the DensePose-Chimps model
    config_file = "densepose_rcnn_R_50_FPN_s1x.yaml"
    model_url = "https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl"
    
    # Try to find config file
    import densepose
    densepose_dir = Path(densepose.__file__).parent
    config_path = densepose_dir / "configs" / config_file
    
    if not config_path.exists():
        # Try alternative locations
        possible_paths = [
            Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / config_file,
            Path("/tmp") / config_file,
        ]
        for p in possible_paths:
            if p.exists():
                config_path = p
                break
        else:
            # Download config
            print(f"Config not found locally, using base config...")
            # Set config manually
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    else:
        cfg.merge_from_file(str(config_path))
    
    cfg.MODEL.WEIGHTS = model_url
    cfg.MODEL.DEVICE = "cpu" if use_cpu or not torch.cuda.is_available() else "cuda"
    
    print(f"Running on: {cfg.MODEL.DEVICE}")
    print(f"Loading model from: {model_url}")
    
    # Load image
    image_path = Path(image_path)
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"ERROR: Could not load image: {image_path}")
        return None
    
    print(f"Image loaded: {image.shape}")
    
    # Run inference
    print("Running inference...")
    predictor = DefaultPredictor(cfg)
    
    with torch.no_grad():
        outputs = predictor(image)
    
    # Extract results
    instances = outputs["instances"]
    print(f"Detected {len(instances)} instances")
    
    if len(instances) == 0:
        print("No instances detected! The model may not recognize chimps.")
        print("Consider using DensePose-Chimps model instead.")
        return None
    
    # Get DensePose predictions
    # The pred_densepose field contains the dense predictions
    if hasattr(instances, 'pred_densepose'):
        print("Found pred_densepose!")
        
        # Extract IUV for first instance
        extractor = DensePoseResultExtractor()
        densepose_results = extractor(instances)
        
        print(f"DensePose results type: {type(densepose_results)}")
        
        # Save results
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Save the raw predictions
        result_data = {
            'image_path': str(image_path),
            'num_instances': len(instances),
            'boxes': instances.pred_boxes.tensor.cpu().numpy(),
        }
        
        # Get the dense IUV arrays
        for i, dp_result in enumerate(densepose_results[0]):
            print(f"\nInstance {i}:")
            print(f"  Type: {type(dp_result)}")
            if hasattr(dp_result, 'labels'):
                labels = dp_result.labels.cpu().numpy()
                print(f"  Labels (I) shape: {labels.shape}, range: [{labels.min()}, {labels.max()}]")
                np.save(output_dir / f"I_instance_{i}.npy", labels)
            if hasattr(dp_result, 'uv'):
                uv = dp_result.uv.cpu().numpy()
                print(f"  UV shape: {uv.shape}")
                np.save(output_dir / f"U_instance_{i}.npy", uv[0])
                np.save(output_dir / f"V_instance_{i}.npy", uv[1])
        
        print(f"\nSaved IUV arrays to {output_dir}")
        return densepose_results
    else:
        print("No pred_densepose in outputs!")
        print(f"Available fields: {instances.get_fields().keys()}")
        return None


def run_densepose_chimps(image_path, output_dir, use_cpu=False):
    """
    Run DensePose-Chimps model specifically designed for chimpanzees.
    
    Saves:
        - I_instance_0.npy: Body part labels (H, W), values 0-14
        - U_instance_0.npy: U coordinates (H, W), values 0-1
        - V_instance_0.npy: V coordinates (H, W), values 0-1
        - bbox_instance_0.npy: Bounding box [x1, y1, x2, y2]
        - visualization.png: Rendered visualization
    """
    import torch
    import cv2
    import matplotlib.pyplot as plt
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    from densepose import add_densepose_config
    from densepose.structures import DensePoseChartPredictorOutput
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Setup output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup config for DensePose-Chimps (CSE model)
    cfg = get_cfg()
    add_densepose_config(cfg)
    
    # DensePose-Chimps model URL
    model_url = "https://dl.fbaipublicfiles.com/densepose/cse/densepose_rcnn_R_50_FPN_soft_chimps_finetune_4k/253146869/model_final_52f649.pkl"
    
    # Config settings
    cfg.MODEL.WEIGHTS = model_url
    cfg.MODEL.DEVICE = "cpu" if use_cpu or not torch.cuda.is_available() else "cuda"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    
    # These are needed for the chimp model
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    
    print(f"Running on: {cfg.MODEL.DEVICE}")
    print(f"Loading DensePose-Chimps model...")
    
    # Load image
    image_path = Path(image_path)
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"ERROR: Could not load image: {image_path}")
        return None
    
    h, w = image.shape[:2]
    print(f"Image loaded: {w}x{h}")
    
    # Run inference
    print("Running inference...")
    try:
        predictor = DefaultPredictor(cfg)
        with torch.no_grad():
            outputs = predictor(image)
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    instances = outputs["instances"].to("cpu")
    print(f"Detected {len(instances)} instances")
    
    if len(instances) == 0:
        print("No chimps detected in image!")
        return None
    
    # Process each detected instance
    for i in range(len(instances)):
        print(f"\nProcessing instance {i}...")
        
        # Get bounding box
        bbox = instances.pred_boxes.tensor[i].numpy()
        x1, y1, x2, y2 = bbox.astype(int)
        print(f"  Bounding box: [{x1}, {y1}, {x2}, {y2}]")
        np.save(output_dir / f"bbox_instance_{i}.npy", bbox)
        
        # Get DensePose predictions
        if hasattr(instances, 'pred_densepose'):
            dp = instances.pred_densepose[i]
            
            # DensePose predictions are in the bounding box coordinate system
            # We need to extract I, U, V
            if isinstance(dp, DensePoseChartPredictorOutput):
                # Get the labels (body part indices) and UV coordinates
                # These are predicted within the bounding box region
                
                # labels: (1, H_box, W_box) - body part indices
                # uv: (2, H_box, W_box) - U and V coordinates
                labels = dp.labels.numpy()  # Shape: (H_box, W_box)
                uv = dp.uv.numpy()  # Shape: (2, H_box, W_box)
                
                print(f"  Labels (I) shape: {labels.shape}, unique values: {np.unique(labels)}")
                print(f"  UV shape: {uv.shape}, U range: [{uv[0].min():.3f}, {uv[0].max():.3f}], V range: [{uv[1].min():.3f}, {uv[1].max():.3f}]")
                
                # Save the raw predictions (in bounding box coordinates)
                np.save(output_dir / f"I_instance_{i}.npy", labels)
                np.save(output_dir / f"U_instance_{i}.npy", uv[0])
                np.save(output_dir / f"V_instance_{i}.npy", uv[1])
                
                # Also create a full-image version with the predictions placed in context
                I_full = np.zeros((h, w), dtype=np.uint8)
                U_full = np.zeros((h, w), dtype=np.float32)
                V_full = np.zeros((h, w), dtype=np.float32)
                
                # Resize predictions to bounding box size and place in full image
                box_h, box_w = y2 - y1, x2 - x1
                if box_h > 0 and box_w > 0:
                    I_resized = cv2.resize(labels.astype(np.float32), (box_w, box_h), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
                    U_resized = cv2.resize(uv[0], (box_w, box_h), interpolation=cv2.INTER_LINEAR)
                    V_resized = cv2.resize(uv[1], (box_w, box_h), interpolation=cv2.INTER_LINEAR)
                    
                    # Clip to image bounds
                    y1c, y2c = max(0, y1), min(h, y2)
                    x1c, x2c = max(0, x1), min(w, x2)
                    
                    # Source region in resized arrays
                    sy1, sy2 = y1c - y1, box_h - (y2 - y2c)
                    sx1, sx2 = x1c - x1, box_w - (x2 - x2c)
                    
                    I_full[y1c:y2c, x1c:x2c] = I_resized[sy1:sy2, sx1:sx2]
                    U_full[y1c:y2c, x1c:x2c] = U_resized[sy1:sy2, sx1:sx2]
                    V_full[y1c:y2c, x1c:x2c] = V_resized[sy1:sy2, sx1:sx2]
                
                np.save(output_dir / f"I_full_instance_{i}.npy", I_full)
                np.save(output_dir / f"U_full_instance_{i}.npy", U_full)
                np.save(output_dir / f"V_full_instance_{i}.npy", V_full)
                
            else:
                print(f"  Unexpected DensePose result type: {type(dp)}")
        else:
            print("  No pred_densepose field!")
            print(f"  Available fields: {instances.get_fields().keys()}")
    
    # Create visualization
    print("\nCreating visualization...")
    create_visualization(image, output_dir, output_dir / "visualization.png")
    
    print(f"\n✓ Results saved to: {output_dir}")
    print(f"  Files:")
    for f in sorted(output_dir.glob("*")):
        size = f.stat().st_size
        print(f"    {f.name}: {size / 1024:.1f} KB")
    
    return outputs, image


def create_visualization(image, data_dir, output_path):
    """Create 4-panel visualization figure."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    import matplotlib.patches as mpatches
    import cv2
    
    data_dir = Path(data_dir)
    
    # Load the saved arrays
    I_full = np.load(data_dir / "I_full_instance_0.npy")
    U_full = np.load(data_dir / "U_full_instance_0.npy")
    V_full = np.load(data_dir / "V_full_instance_0.npy")
    
    # Build texture atlas
    atlas, coverage = build_texture_atlas_from_arrays(image, I_full, U_full, V_full)
    
    # Save atlas and coverage
    np.save(data_dir / "texture_atlas.npy", atlas)
    np.save(data_dir / "coverage_map.npy", coverage)
    
    # Create figure
    part_names = ['BG', 'Torso', 'RHand', 'LHand', 'RFoot', 'LFoot',
                  'RUpLeg', 'LUpLeg', 'RLoLeg', 'LLoLeg',
                  'RUpArm', 'LUpArm', 'RLoArm', 'LLoArm', 'Head']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    
    # A: Original image (convert BGR to RGB)
    axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('A. Original Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # B: Body part segmentation
    colors = plt.cm.tab20(np.linspace(0, 1, 15))
    colors[0] = [0, 0, 0, 1]  # Background = black
    cmap = ListedColormap(colors)
    axes[0, 1].imshow(I_full, cmap=cmap, vmin=0, vmax=14)
    axes[0, 1].set_title('B. Body Part Segmentation (I)', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Legend
    patches = [mpatches.Patch(color=colors[i], label=part_names[i]) 
               for i in range(1, 15) if i in np.unique(I_full)]
    if patches:
        axes[0, 1].legend(handles=patches, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
    
    # C: Surface atlas
    axes[1, 0].imshow(atlas)
    axes[1, 0].set_title('C. Surface Atlas (UV Texture Map)', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Grid lines
    cell_size = atlas.shape[0] // 4
    for i in range(1, 4):
        axes[1, 0].axhline(y=i*cell_size, color='white', linewidth=1, alpha=0.7)
        axes[1, 0].axvline(x=i*cell_size, color='white', linewidth=1, alpha=0.7)
    
    # D: Coverage map
    im = axes[1, 1].imshow(coverage, cmap='YlOrRd', vmin=0, vmax=1)
    axes[1, 1].set_title('D. Coverage Map', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    plt.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04, label='Observed')
    
    for i in range(1, 4):
        axes[1, 1].axhline(y=i*cell_size, color='gray', linewidth=1, alpha=0.5)
        axes[1, 1].axvline(x=i*cell_size, color='gray', linewidth=1, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved visualization to: {output_path}")


def build_texture_atlas_from_arrays(image, I_map, U_map, V_map, atlas_size=512):
    """
    Build UV texture atlas from DensePose predictions.
    """
    import cv2
    
    # Convert BGR to RGB
    if len(image.shape) == 3 and image.shape[2] == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image
    
    grid_size = 4
    cell_size = atlas_size // grid_size
    
    atlas = np.zeros((atlas_size, atlas_size, 3), dtype=np.uint8)
    coverage = np.zeros((atlas_size, atlas_size), dtype=np.float32)
    
    h, w = I_map.shape
    
    for y in range(h):
        for x in range(w):
            part_id = int(I_map[y, x])
            if part_id == 0:
                continue
            
            u = np.clip(float(U_map[y, x]), 0, 1)
            v = np.clip(float(V_map[y, x]), 0, 1)
            
            cell_row = (part_id - 1) // grid_size
            cell_col = (part_id - 1) % grid_size
            
            atlas_x = int(cell_col * cell_size + u * (cell_size - 1))
            atlas_y = int(cell_row * cell_size + v * (cell_size - 1))
            
            if 0 <= atlas_x < atlas_size and 0 <= atlas_y < atlas_size:
                atlas[atlas_y, atlas_x] = image_rgb[y, x]
                coverage[atlas_y, atlas_x] = 1.0
    
    return atlas, coverage


def simple_test():
    """
    Simple test to check if detectron2 and densepose are properly installed.
    """
    print("Testing detectron2 and densepose installation...")
    
    try:
        import detectron2
        print(f"✓ detectron2 version: {detectron2.__version__}")
    except ImportError as e:
        print(f"✗ detectron2 import failed: {e}")
        return False
    
    try:
        import densepose
        print(f"✓ densepose imported from: {densepose.__file__}")
    except ImportError as e:
        print(f"✗ densepose import failed: {e}")
        return False
    
    try:
        from densepose import add_densepose_config
        print("✓ add_densepose_config available")
    except ImportError as e:
        print(f"✗ add_densepose_config import failed: {e}")
        return False
    
    try:
        from densepose.vis.extractor import DensePoseResultExtractor
        print("✓ DensePoseResultExtractor available")
    except ImportError as e:
        print(f"✗ DensePoseResultExtractor import failed: {e}")
    
    # Check what's in densepose
    print(f"\nDensepose contents: {[x for x in dir(densepose) if not x.startswith('_')]}")
    
    # Check for configs
    densepose_dir = Path(densepose.__file__).parent
    print(f"\nDensepose directory: {densepose_dir}")
    if (densepose_dir / "configs").exists():
        configs = list((densepose_dir / "configs").glob("*.yaml"))
        print(f"Found {len(configs)} config files")
        for c in configs[:5]:
            print(f"  - {c.name}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Run DensePose on a single image')
    parser.add_argument('image', nargs='?', help='Path to image file')
    parser.add_argument('--output-dir', '-o', default='output', help='Output directory')
    parser.add_argument('--cpu', action='store_true', help='Force CPU mode')
    parser.add_argument('--test', action='store_true', help='Just test if densepose is installed')
    parser.add_argument('--chimps', action='store_true', help='Use DensePose-Chimps model')
    args = parser.parse_args()
    
    if args.test:
        simple_test()
        return
    
    if not args.image:
        print("Usage: python run_densepose_image.py <image_path>")
        print("       python run_densepose_image.py --test")
        return
    
    if args.chimps:
        run_densepose_chimps(args.image, args.output_dir, args.cpu)
    else:
        run_densepose(args.image, args.output_dir, args.cpu)


if __name__ == "__main__":
    main()
