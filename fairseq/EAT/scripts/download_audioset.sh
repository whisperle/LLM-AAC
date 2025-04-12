#!/bin/bash

# Set destination directory
OUTPUT_DIR="/scratch/cl6707/Shared_Datasets/AudioSet"

# Default to downloading both configurations
CONFIGS="balanced,unbalanced"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --configs)
      CONFIGS="$2"
      shift 2
      ;;
    --balanced-only)
      CONFIGS="balanced"
      shift
      ;;
    --unbalanced-only)
      CONFIGS="unbalanced"
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Create directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo "Starting AudioSet download:"
echo "  Destination: $OUTPUT_DIR"
echo "  Configurations: $CONFIGS"

# Check if tqdm is installed
python -c "import tqdm" > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "Installing tqdm for progress display..."
    pip install tqdm
fi

# Run the Python downloader script
cd "$(dirname "$0")/.."
python scripts/download_audioset.py --output_dir "$OUTPUT_DIR" --configs "$CONFIGS"

# Make sure permissions are set correctly
chmod -R 755 "$OUTPUT_DIR"

echo "Download process completed!" 