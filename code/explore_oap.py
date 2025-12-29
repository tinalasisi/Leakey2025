#!/usr/bin/env python3
"""
Explore species in OpenApePose dataset
"""
import json
from collections import Counter

def explore_dataset(ann_file):
    with open(ann_file) as f:
        data = json.load(f)
    
    if isinstance(data, dict) and 'data' in data:
        annotations = data['data']
    else:
        annotations = data
    
    # Count species
    species_counts = Counter(ann['species'] for ann in annotations)
    
    print(f"Total images: {len(annotations)}")
    print(f"\nSpecies breakdown ({len(species_counts)} species):")
    print("-" * 40)
    
    for species, count in species_counts.most_common():
        # Also count how many have good visibility (15+ landmarks)
        good_vis = sum(1 for ann in annotations 
                      if ann['species'] == species and sum(ann['visibility']) >= 15)
        print(f"{species:30s}: {count:6d} total, {good_vis:5d} with 15+ landmarks")

if __name__ == '__main__':
    import sys
    ann_file = sys.argv[1] if len(sys.argv) > 1 else 'oap_all.json'
    explore_dataset(ann_file)
