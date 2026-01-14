#!/bin/bash
# Download OpenApePose dataset from Dryad using API v2
# Dataset: https://doi.org/10.5061/dryad.c59zw3rds
# Total size: ~175 GB (8 split 7z archives)

# Target directory on turbo storage
TARGET_DIR="/nfs/turbo/lsa-tlasisi1/tlasisi/Leakey2025/data/OpenApePose"
mkdir -p "$TARGET_DIR"
cd "$TARGET_DIR"

echo "Downloading OpenApePose dataset to: $TARGET_DIR"
echo "Started at: $(date)"

# Download using Dryad API v2 endpoints
# File IDs from: https://datadryad.org/api/v2/versions/248198/files

# README (843 bytes)
echo "Downloading README.MD..."
curl -L -o README.MD "https://datadryad.org/api/v2/files/2468008/download"

# Split 7z archives (~25 GB each, last one ~219 MB)
for i in $(seq 1 8); do
    FILEID=$((2467998 + i))  # IDs are 2467999-2468006
    FILENAME=$(printf "OpenApePose.7z.%03d" $i)
    
    if [ -f "$FILENAME" ]; then
        echo "$FILENAME already exists, skipping..."
        continue
    fi
    
    echo "Downloading $FILENAME (file ID: $FILEID)..."
    curl -L -C - -o "$FILENAME" "https://datadryad.org/api/v2/files/${FILEID}/download"
    
    if [ $? -eq 0 ]; then
        echo "$FILENAME complete"
    else
        echo "ERROR downloading $FILENAME"
    fi
done

echo ""
echo "Finished at: $(date)"
echo ""
echo "To extract, run:"
echo "  cd $TARGET_DIR"
echo "  7z x OpenApePose.7z.001"