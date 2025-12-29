#!/usr/bin/env python3
"""
DensePose Surface Atlas Explorer

Run this from your project directory. It will:
1. Create an 'output' subdirectory for results
2. Explore what Zamba provides
3. Build a texture atlas if dense IUV data is available

Usage:
    cd /path/to/your/project
    python explore_zamba_densepose.py

Or with a specific image:
    python explore_zamba_densepose.py --image path/to/chimp.jpg

Author: Tina Lasisi
"""

import os
import sys
import json
import argparse
import subprocess
import numpy as np
from pathlib import Path
from datetime import datetime

# ============================================================
# UTILITIES
# ============================================================

def human_readable_size(size_bytes):
    """Convert bytes to human readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"

def print_directory_summary(directory):
    """Print summary of files in a directory."""
    directory = Path(directory)
    if not directory.exists():
        print(f"Directory does not exist: {directory}")
        return
    
    files = list(directory.iterdir())
    if not files:
        print(f"Directory is empty: {directory}")
        return
    
    print(f"\n{'='*60}")
    print(f"OUTPUT SUMMARY: {directory}")
    print(f"{'='*60}")
    print(f"Total files: {len(files)}")
    print(f"\n{'File':<40} {'Size':>12} {'Type':<10}")
    print(f"{'-'*40} {'-'*12} {'-'*10}")
    
    total_size = 0
    by_extension = {}
    
    for f in sorted(files):
        if f.is_file():
            size = f.stat().st_size
            total_size += size
            ext = f.suffix or '(none)'
            by_extension[ext] = by_extension.get(ext, 0) + 1
            print(f"{f.name:<40} {human_readable_size(size):>12} {ext:<10}")
    
    print(f"{'-'*40} {'-'*12} {'-'*10}")
    print(f"{'TOTAL':<40} {human_readable_size(total_size):>12}")
    print(f"\nBy type: {dict(by_extension)}")

def setup_project_directories():
    """
    Create output directory in project folder.
    
    Assumes script is in project/code/ and creates:
        project/output/
        project/data/
    """
    # Get the directory containing this script
    script_dir = Path(__file__).parent.resolve()
    
    # Project root is parent of code/ directory
    if script_dir.name == 'code':
        project_dir = script_dir.parent
    else:
        # Script is in project root or elsewhere
        project_dir = Path.cwd()
    
    output_dir = project_dir / "output"
    data_dir = project_dir / "data"
    
    output_dir.mkdir(exist_ok=True)
    data_dir.mkdir(exist_ok=True)
    
    print(f"Project directory: {project_dir}")
    print(f"Code directory:    {script_dir}")
    print(f"Output directory:  {output_dir}")
    print(f"Data directory:    {data_dir}")
    
    return {
        'project': project_dir,
        'code': script_dir,
        'output': output_dir,
        'data': data_dir,
    }

# ============================================================
# TEXTURE ATLAS BUILDING
# ============================================================

def build_texture_atlas(image, I_map, U_map, V_map, atlas_size=512):
    """
    Build UV texture atlas from DensePose predictions.
    
    Args:
        image: Original RGB image as numpy array (H, W, 3)
        I_map: Body part indices (H, W), 0=background, 1-14=parts
        U_map: U coordinates (H, W), range 0-1
        V_map: V coordinates (H, W), range 0-1
        atlas_size: Output atlas size in pixels
    
    Returns:
        atlas: RGB texture atlas (atlas_size, atlas_size, 3)
        coverage: Coverage mask (atlas_size, atlas_size)
    """
    import numpy as np
    
    # 14 body parts arranged in 4x4 grid
    grid_size = 4
    cell_size = atlas_size // grid_size
    
    atlas = np.zeros((atlas_size, atlas_size, 3), dtype=np.uint8)
    coverage = np.zeros((atlas_size, atlas_size), dtype=np.float32)
    
    h, w = I_map.shape
    
    for y in range(h):
        for x in range(w):
            part_id = int(I_map[y, x])
            if part_id == 0:  # Skip background
                continue
            
            u = np.clip(float(U_map[y, x]), 0, 1)
            v = np.clip(float(V_map[y, x]), 0, 1)
            
            # Map to grid cell (part 1-14 -> cells 0-13)
            cell_row = (part_id - 1) // grid_size
            cell_col = (part_id - 1) % grid_size
            
            # Position within cell
            atlas_x = int(cell_col * cell_size + u * (cell_size - 1))
            atlas_y = int(cell_row * cell_size + v * (cell_size - 1))
            
            if 0 <= atlas_x < atlas_size and 0 <= atlas_y < atlas_size:
                atlas[atlas_y, atlas_x] = image[y, x]
                coverage[atlas_y, atlas_x] = 1.0
    
    return atlas, coverage

def create_rebuttal_figure(image, I_map, atlas, coverage, output_path):
    """
    Create 4-panel figure for grant rebuttal.
    
    Args:
        image: Original image (H, W, 3)
        I_map: Body part segmentation (H, W)
        atlas: Texture atlas (atlas_size, atlas_size, 3)
        coverage: Coverage map (atlas_size, atlas_size)
        output_path: Where to save the figure
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    import matplotlib.patches as mpatches
    
    part_names = ['BG', 'Torso', 'RHand', 'LHand', 'RFoot', 'LFoot',
                  'RUpLeg', 'LUpLeg', 'RLoLeg', 'LLoLeg',
                  'RUpArm', 'LUpArm', 'RLoArm', 'LLoArm', 'Head']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    
    # A: Original
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('A. Original Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # B: Body parts (I values)
    colors = plt.cm.tab20(np.linspace(0, 1, 15))
    colors[0] = [0, 0, 0, 1]  # Background = black
    cmap = ListedColormap(colors)
    im = axes[0, 1].imshow(I_map, cmap=cmap, vmin=0, vmax=14)
    axes[0, 1].set_title('B. Body Part Segmentation', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Add legend
    patches = [mpatches.Patch(color=colors[i], label=part_names[i]) 
               for i in range(1, min(15, len(part_names)))]
    axes[0, 1].legend(handles=patches, loc='center left', bbox_to_anchor=(1, 0.5),
                       fontsize=8, ncol=1)
    
    # C: Surface atlas
    axes[1, 0].imshow(atlas)
    axes[1, 0].set_title('C. Surface Atlas (UV Texture Map)', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Add grid lines
    cell_size = atlas.shape[0] // 4
    for i in range(1, 4):
        axes[1, 0].axhline(y=i*cell_size, color='white', linewidth=1, alpha=0.7)
        axes[1, 0].axvline(x=i*cell_size, color='white', linewidth=1, alpha=0.7)
    
    # D: Coverage
    im = axes[1, 1].imshow(coverage, cmap='YlOrRd', vmin=0, vmax=1)
    axes[1, 1].set_title('D. Coverage Map', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    plt.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04, label='Observed')
    
    # Grid lines on coverage
    for i in range(1, 4):
        axes[1, 1].axhline(y=i*cell_size, color='gray', linewidth=1, alpha=0.5)
        axes[1, 1].axvline(x=i*cell_size, color='gray', linewidth=1, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved rebuttal figure to: {output_path}")
    print(f"  Size: {human_readable_size(Path(output_path).stat().st_size)}")

# ============================================================
# ZAMBA EXPLORATION
# ============================================================

def check_zamba_installation():
    """Check if Zamba is installed and accessible."""
    print("\n" + "="*60)
    print("CHECKING ZAMBA INSTALLATION")
    print("="*60)
    
    # Check CLI
    try:
        result = subprocess.run(['zamba', '--version'], capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print(f"✓ Zamba CLI available: {result.stdout.strip()}")
        else:
            # Try --help instead
            result = subprocess.run(['zamba', '--help'], capture_output=True, text=True, timeout=30)
            if 'zamba' in result.stdout.lower():
                print(f"✓ Zamba CLI available")
            else:
                print(f"✗ Zamba CLI issue: {result.stderr[:200]}")
                return False
    except FileNotFoundError:
        print("✗ Zamba CLI not found in PATH")
        return False
    except Exception as e:
        print(f"✗ Error checking Zamba: {e}")
        return False
    
    # Check densepose subcommand
    try:
        result = subprocess.run(['zamba', 'densepose', '--help'], capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print(f"✓ Zamba densepose subcommand available")
            # Extract key options
            help_text = result.stdout
            if '--output-type' in help_text:
                print(f"  --output-type options available")
            if '--render-output' in help_text:
                print(f"  --render-output flag available")
        else:
            print(f"✗ Zamba densepose not available: {result.stderr[:200]}")
            return False
    except Exception as e:
        print(f"✗ Error checking densepose: {e}")
        return False
    
    return True

def run_zamba_densepose(images_dir, output_dir, output_type='chimp_anatomy', render=True):
    """
    Run Zamba DensePose on images in a directory.
    
    Returns path to output directory.
    """
    print("\n" + "="*60)
    print(f"RUNNING ZAMBA DENSEPOSE")
    print("="*60)
    
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    
    # Check for images
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpeg"))
    if not image_files:
        print(f"✗ No images found in {images_dir}")
        print(f"  Please add .jpg or .png files to: {images_dir}")
        print(f"  Or use --data-dir to specify a different directory")
        return None
    
    print(f"Found {len(image_files)} image(s):")
    for img in image_files[:5]:  # Show first 5
        print(f"  - {img.name}")
    if len(image_files) > 5:
        print(f"  ... and {len(image_files) - 5} more")
    
    # Build command
    cmd = [
        'zamba', 'densepose',
        '--data-dir', str(images_dir),
        '--save-dir', str(output_dir),
        '--output-type', output_type,
    ]
    if render:
        cmd.append('--render-output')
    
    print(f"\nCommand: {' '.join(cmd)}")
    print(f"\nRunning... (this may take a while on CPU, faster on GPU)")
    
    start_time = datetime.now()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)  # 10 min timeout
        
        elapsed = datetime.now() - start_time
        print(f"\nCompleted in {elapsed.total_seconds():.1f} seconds")
        
        if result.returncode == 0:
            print("✓ Zamba completed successfully")
        else:
            print(f"✗ Zamba returned error code {result.returncode}")
            if result.stderr:
                print(f"Stderr: {result.stderr[:500]}")
        
        if result.stdout:
            print(f"\nStdout:\n{result.stdout[:1000]}")
            
    except subprocess.TimeoutExpired:
        print("✗ Timeout after 10 minutes - may need GPU for faster processing")
        return None
    except Exception as e:
        print(f"✗ Error running Zamba: {e}")
        return None
    
    return output_dir

def inspect_zamba_output(output_dir):
    """
    Inspect Zamba output files to understand what data is available.
    """
    print("\n" + "="*60)
    print("INSPECTING ZAMBA OUTPUT")
    print("="*60)
    
    output_dir = Path(output_dir)
    
    # First, print summary
    print_directory_summary(output_dir)
    
    # Now inspect each file type
    print("\n" + "-"*60)
    print("DETAILED FILE INSPECTION")
    print("-"*60)
    
    has_dense_iuv = False
    iuv_files = {}
    
    for f in sorted(output_dir.iterdir()):
        if not f.is_file():
            continue
            
        print(f"\n>>> {f.name}")
        
        if f.suffix == '.json':
            try:
                with open(f) as jf:
                    data = json.load(jf)
                print(f"    Type: JSON")
                if isinstance(data, dict):
                    print(f"    Keys: {list(data.keys())}")
                    for k, v in data.items():
                        if isinstance(v, list):
                            print(f"      {k}: list, len={len(v)}")
                            if len(v) > 0:
                                print(f"        first item type: {type(v[0]).__name__}")
                                if isinstance(v[0], dict):
                                    print(f"        first item keys: {list(v[0].keys())[:10]}")
                        elif isinstance(v, dict):
                            print(f"      {k}: dict, keys={list(v.keys())[:5]}")
                        else:
                            print(f"      {k}: {type(v).__name__} = {str(v)[:50]}")
                elif isinstance(data, list):
                    print(f"    List with {len(data)} items")
                    if len(data) > 0:
                        print(f"    First item: {type(data[0]).__name__}")
            except Exception as e:
                print(f"    Error reading: {e}")
                
        elif f.suffix == '.csv':
            try:
                import pandas as pd
                df = pd.read_csv(f)
                print(f"    Type: CSV")
                print(f"    Shape: {df.shape[0]} rows × {df.shape[1]} columns")
                print(f"    Columns: {list(df.columns)}")
                print(f"    First row:\n{df.head(1).to_string()}")
            except ImportError:
                # Read manually
                with open(f) as cf:
                    lines = cf.readlines()
                print(f"    Type: CSV")
                print(f"    Lines: {len(lines)}")
                if lines:
                    print(f"    Header: {lines[0].strip()}")
                    if len(lines) > 1:
                        print(f"    First data: {lines[1].strip()[:100]}")
            except Exception as e:
                print(f"    Error reading: {e}")
                
        elif f.suffix == '.npy':
            try:
                import numpy as np
                arr = np.load(f)
                print(f"    Type: NumPy array")
                print(f"    Shape: {arr.shape}")
                print(f"    Dtype: {arr.dtype}")
                print(f"    Range: [{arr.min():.3f}, {arr.max():.3f}]")
                
                # Check if this might be I, U, or V
                if len(arr.shape) == 2:
                    h, w = arr.shape
                    if arr.max() <= 14 and arr.min() >= 0 and arr.dtype in [np.int32, np.int64, np.uint8]:
                        print(f"    >>> Likely body part indices (I)")
                        has_dense_iuv = True
                        iuv_files['I'] = f
                    elif arr.max() <= 1.0 and arr.min() >= 0:
                        print(f"    >>> Likely UV coordinates")
                        has_dense_iuv = True
                        if 'U' not in iuv_files:
                            iuv_files['U'] = f
                        else:
                            iuv_files['V'] = f
            except Exception as e:
                print(f"    Error reading: {e}")
                
        elif f.suffix in ['.mp4', '.avi', '.mov']:
            print(f"    Type: Video")
            print(f"    (Rendered visualization)")
            
        elif f.suffix in ['.png', '.jpg', '.jpeg']:
            try:
                from PIL import Image
                img = Image.open(f)
                print(f"    Type: Image")
                print(f"    Size: {img.size}")
                print(f"    Mode: {img.mode}")
            except:
                print(f"    Type: Image (couldn't read details)")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if has_dense_iuv:
        print("✓ Found dense IUV data! Can build texture atlas.")
        print(f"  Files: {iuv_files}")
    else:
        print("✗ No dense IUV arrays found.")
        print("  Zamba may only output summary statistics.")
        print("  Options:")
        print("    1. Check if there's a different --output-type that gives dense output")
        print("    2. Use detectron2 directly (see detectron2_fallback.py)")
    
    return has_dense_iuv, iuv_files

# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Explore Zamba DensePose output')
    parser.add_argument('--image', type=str, help='Path to a specific image to process')
    parser.add_argument('--data-dir', type=str, help='Directory containing images')
    parser.add_argument('--output-type', type=str, default='chimp_anatomy',
                        help='Zamba output type (default: chimp_anatomy)')
    parser.add_argument('--skip-run', action='store_true', 
                        help='Skip running Zamba, just inspect existing output')
    args = parser.parse_args()
    
    print("="*60)
    print("DENSEPOSE SURFACE ATLAS EXPLORER")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # Setup directories in current project folder
    dirs = setup_project_directories()
    
    # If specific image provided, copy to data dir
    if args.image:
        import shutil
        src = Path(args.image)
        if src.exists():
            dst = dirs['data'] / src.name
            shutil.copy(src, dst)
            print(f"Copied {src.name} to data directory")
        else:
            print(f"Image not found: {args.image}")
            return
    
    # Override data dir if specified
    if args.data_dir:
        dirs['data'] = Path(args.data_dir)
    
    # Check installation
    if not check_zamba_installation():
        print("\nZamba not properly installed. Please check your environment.")
        return
    
    # Run Zamba (unless skipping)
    if not args.skip_run:
        result_dir = run_zamba_densepose(
            dirs['data'], 
            dirs['output'],
            output_type=args.output_type
        )
        if result_dir is None:
            print("\nFailed to run Zamba. Check images directory and try again.")
            return
    
    # Inspect output
    has_iuv, iuv_files = inspect_zamba_output(dirs['output'])
    
    # Print final summary
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    
    if has_iuv:
        print("""
✓ Dense IUV data found! You can build the texture atlas.

Run:
    python build_texture_atlas.py --i {I_file} --u {U_file} --v {V_file} --image {image}

Or use the build_texture_atlas() function from the script.
""")
    else:
        print("""
The Zamba output doesn't include dense per-pixel IUV predictions.
This means we need to use detectron2 directly.

Options:
1. Try a different --output-type:
   python explore_zamba_densepose.py --output-type segmentation

2. Use detectron2 directly (requires GPU):
   See GREATLAKES_DENSEPOSE_INSTRUCTIONS.md for setup

3. Check if rendered video/images show the IUV overlay
   (useful for visualization but not for building atlas)
""")
    
    print(f"\nOutput files are in: {dirs['output']}")

if __name__ == "__main__":
    main()