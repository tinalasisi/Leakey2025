#!/bin/bash
# Download OpenApePose dataset from Dryad
# Run this from the Leakey2025 repo root on Great Lakes
# Total size: ~175 GB
#
# Usage:
#   cd /nfs/turbo/lsa-tlasisi1/tlasisi/Leakey2025
#   screen -S download
#   bash code/download_openapepose.sh

# Set up directory (relative to repo root)
mkdir -p data/OpenApePose
cd data/OpenApePose

echo "Starting downloads at $(date)"
echo "Downloading to: $(pwd)"

# Download all 8 parts + README
# Using -c flag to resume if interrupted

wget -c -O OpenApePose.7z.001 "https://datadryad.org/downloads/file_stream/2467999"
echo "Part 1/8 complete"

wget -c -O OpenApePose.7z.002 "https://datadryad.org/downloads/file_stream/2468000"
echo "Part 2/8 complete"

wget -c -O OpenApePose.7z.003 "https://datadryad.org/downloads/file_stream/2468001"
echo "Part 3/8 complete"

wget -c -O OpenApePose.7z.004 "https://datadryad.org/downloads/file_stream/2468002"
echo "Part 4/8 complete"

wget -c -O OpenApePose.7z.005 "https://datadryad.org/downloads/file_stream/2468003"
echo "Part 5/8 complete"

wget -c -O OpenApePose.7z.006 "https://datadryad.org/downloads/file_stream/2468004"
echo "Part 6/8 complete"

wget -c -O OpenApePose.7z.007 "https://datadryad.org/downloads/file_stream/2468005"
echo "Part 7/8 complete"

wget -c -O OpenApePose.7z.008 "https://datadryad.org/downloads/file_stream/2468006"
echo "Part 8/8 complete"

wget -c -O README.MD "https://datadryad.org/downloads/file_stream/2468008"
echo "README complete"

echo "All downloads complete at $(date)"
echo ""
echo "To extract, run:"
echo "  module load p7zip"
echo "  7z x OpenApePose.7z.001"