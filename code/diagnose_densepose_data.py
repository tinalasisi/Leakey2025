#!/usr/bin/env python3
"""
Diagnose the DensePose pkl data structure.
Let's see exactly what we have before trying to visualize it.
"""

import numpy as np
import torch
from pathlib import Path

PKL_PATH = Path("/nfs/turbo/lsa-tlasisi1/tlasisi/Leakey2025/output/densepose_iuv_results.pkl")

print("="*70)
print("DENSEPOSE DATA DIAGNOSTIC")
print("="*70)

# Load
results = torch.load(PKL_PATH, map_location='cpu', weights_only=False)

print(f"\n1. TOP-LEVEL STRUCTURE")
print(f"   Type: {type(results)}")
print(f"   Length: {len(results)}")

print(f"\n2. FIRST RESULT KEYS")
result = results[0]
print(f"   Type: {type(result)}")
if isinstance(result, dict):
    for key in result.keys():
        val = result[key]
        print(f"   '{key}': {type(val)}")
        if hasattr(val, 'shape'):
            print(f"       shape: {val.shape}")
        elif hasattr(val, '__len__'):
            print(f"       len: {len(val)}")

print(f"\n3. BOUNDING BOX")
boxes = result.get('pred_boxes_XYXY', None)
if boxes is not None:
    if hasattr(boxes, 'cpu'):
        boxes = boxes.cpu().numpy()
    print(f"   Shape: {boxes.shape}")
    print(f"   Values: {boxes}")

print(f"\n4. DENSEPOSE PREDICTIONS")
dp_list = result.get('pred_densepose', [])
print(f"   Number of instances: {len(dp_list)}")

if len(dp_list) > 0:
    dp = dp_list[0]
    print(f"\n   Instance 0:")
    print(f"   Type: {type(dp)}")
    print(f"   Attributes: {[a for a in dir(dp) if not a.startswith('_')]}")
    
    # Check for labels (I)
    if hasattr(dp, 'labels'):
        labels = dp.labels
        if hasattr(labels, 'cpu'):
            labels = labels.cpu().numpy()
        print(f"\n5. LABELS (I - body parts)")
        print(f"   Shape: {labels.shape}")
        print(f"   Dtype: {labels.dtype}")
        print(f"   Min: {labels.min()}, Max: {labels.max()}")
        unique, counts = np.unique(labels, return_counts=True)
        print(f"   Unique values and counts:")
        for u, c in zip(unique, counts):
            pct = c / labels.size * 100
            print(f"      Part {u:2d}: {c:6d} pixels ({pct:5.1f}%)")
        
        # Non-background pixels
        body_pixels = np.sum(labels > 0)
        print(f"\n   Total body pixels (I > 0): {body_pixels}")
    
    # Check for UV
    if hasattr(dp, 'uv'):
        uv = dp.uv
        if hasattr(uv, 'cpu'):
            uv = uv.cpu().numpy()
        print(f"\n6. UV COORDINATES")
        print(f"   Shape: {uv.shape}")
        print(f"   Dtype: {uv.dtype}")
        
        u = uv[0]
        v = uv[1]
        print(f"\n   U channel:")
        print(f"      Shape: {u.shape}")
        print(f"      Min: {u.min():.4f}, Max: {u.max():.4f}")
        print(f"      Mean: {u.mean():.4f}, Std: {u.std():.4f}")
        
        print(f"\n   V channel:")
        print(f"      Shape: {v.shape}")
        print(f"      Min: {v.min():.4f}, Max: {v.max():.4f}")
        print(f"      Mean: {v.mean():.4f}, Std: {v.std():.4f}")
        
        # Sample some actual values where we have body pixels
        if hasattr(dp, 'labels'):
            labels_np = dp.labels.cpu().numpy() if hasattr(dp.labels, 'cpu') else dp.labels
            body_mask = labels_np > 0
            print(f"\n   Sample U,V values at body pixels:")
            body_indices = np.where(body_mask)
            for i in range(min(10, len(body_indices[0]))):
                y, x = body_indices[0][i], body_indices[1][i]
                print(f"      Pixel ({x:3d},{y:3d}): I={labels_np[y,x]:2d}, U={u[y,x]:.3f}, V={v[y,x]:.3f}")
    
    # Check for other attributes (CSE embeddings, confidence, etc.)
    if hasattr(dp, 'embedding'):
        emb = dp.embedding
        if hasattr(emb, 'shape'):
            print(f"\n7. EMBEDDINGS (CSE)")
            print(f"   Shape: {emb.shape}")
    
    if hasattr(dp, 'coarse_segm'):
        cs = dp.coarse_segm
        if hasattr(cs, 'cpu'):
            cs = cs.cpu().numpy()
        print(f"\n8. COARSE SEGMENTATION")
        print(f"   Shape: {cs.shape}")

print("\n" + "="*70)
print("END DIAGNOSTIC")
print("="*70)
