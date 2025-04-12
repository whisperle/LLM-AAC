#!/bin/bash

# Script to pre-compute CWT features for AudioSet and calculate normalization statistics

# Set base directory for output
OUTPUT_DIR="/scratch/cl6707/Shared_Datasets/AudioSet_Wavelets"

# Default parameters
CONFIG="balanced"
SPLIT="train"
CWT_SCALES=64
CWT_WIDTH=8.0
TARGET_LENGTH=1024
NUM_WORKERS=0  # 0 means auto-detect (CPU count - 1)
CHUNK_SIZE=200
MAX_SAMPLES=0  # 0 means process all samples

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --split)
      SPLIT="$2"
      shift 2
      ;;
    --cwt_scales)
      CWT_SCALES="$2"
      shift 2
      ;;
    --cwt_width)
      CWT_WIDTH="$2"
      shift 2
      ;;
    --target_length)
      TARGET_LENGTH="$2"
      shift 2
      ;;
    --num_workers)
      NUM_WORKERS="$2"
      shift 2
      ;;
    --chunk_size)
      CHUNK_SIZE="$2"
      shift 2
      ;;
    --max_samples)
      MAX_SAMPLES="$2"
      shift 2
      ;;
    --balanced)
      CONFIG="balanced"
      shift
      ;;
    --unbalanced)
      CONFIG="unbalanced"
      shift
      ;;
    --both-configs)
      # Will run both configs sequentially
      RUN_BOTH_CONFIGS=1
      shift
      ;;
    --help)
      echo "Usage: $(basename "$0") [options]"
      echo ""
      echo "Options:"
      echo "  --output_dir DIR         Directory to save pre-computed features (default: $OUTPUT_DIR)"
      echo "  --config CONFIG          AudioSet configuration: balanced or unbalanced (default: $CONFIG)"
      echo "  --split SPLIT            Dataset split: train or test (default: $SPLIT)"
      echo "  --cwt_scales N           Number of scales for CWT (default: $CWT_SCALES)"
      echo "  --cwt_width N            Width parameter for Morlet wavelet (default: $CWT_WIDTH)"
      echo "  --target_length N        Target length for time dimension (default: $TARGET_LENGTH)"
      echo "  --num_workers N          Number of worker processes (default: auto)"
      echo "  --chunk_size N           Number of samples per chunk (default: $CHUNK_SIZE)"
      echo "  --max_samples N          Maximum samples to process, 0 for all (default: $MAX_SAMPLES)"
      echo "  --balanced               Use balanced config (shorthand)"
      echo "  --unbalanced             Use unbalanced config (shorthand)"
      echo "  --both-configs           Process both balanced and unbalanced configs"
      echo "  --help                   Show this help message and exit"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Function to run the Python script with given parameters
run_precompute() {
    local config=$1
    local split=$2
    
    echo "================================================================="
    echo "Processing AudioSet $config - $split split"
    echo "Output directory: $OUTPUT_DIR"
    echo "CWT scales: $CWT_SCALES, width: $CWT_WIDTH"
    echo "================================================================="
    
    # Check for packages and install if missing
    for pkg in tqdm h5py pywt scipy; do
        if ! python -c "import $pkg" &>/dev/null; then
            echo "Installing $pkg..."
            pip install "$pkg"
        fi
    done
    
    # Run the pre-computation script
    cmd="python $(dirname "$0")/precompute_wavelets.py \
        --output_dir $OUTPUT_DIR \
        --config $config \
        --split $split \
        --cwt_scales $CWT_SCALES \
        --cwt_width $CWT_WIDTH \
        --target_length $TARGET_LENGTH \
        --chunk_size $CHUNK_SIZE"
        
    # Add optional arguments if specified
    if [ "$NUM_WORKERS" -gt 0 ]; then
        cmd="$cmd --num_workers $NUM_WORKERS"
    fi
    
    if [ "$MAX_SAMPLES" -gt 0 ]; then
        cmd="$cmd --max_samples $MAX_SAMPLES"
    fi
    
    echo "Running command: $cmd"
    echo ""
    
    # Execute the command
    eval "$cmd"
    
    # Check if successful
    if [ $? -eq 0 ]; then
        echo "Successfully completed pre-computation for $config-$split!"
    else
        echo "Error: Pre-computation failed for $config-$split."
        return 1
    fi
    
    return 0
}

# Run for specified configuration or both
if [ -n "$RUN_BOTH_CONFIGS" ]; then
    # Process balanced first
    run_precompute "balanced" "$SPLIT"
    bal_result=$?
    
    # Then process unbalanced
    run_precompute "unbalanced" "$SPLIT"
    unbal_result=$?
    
    # Check results
    if [ $bal_result -eq 0 ] && [ $unbal_result -eq 0 ]; then
        echo "All pre-computations completed successfully!"
    else
        echo "Warning: Some pre-computations failed."
        exit 1
    fi
else
    # Process single configuration
    run_precompute "$CONFIG" "$SPLIT"
    if [ $? -ne 0 ]; then
        exit 1
    fi
fi

echo ""
echo "Pre-computation complete! Features saved to: $OUTPUT_DIR"
echo "Remember to update your dataset class to use these pre-computed features." 