#!/bin/bash

# Script for pre-training EAT model using pre-computed CWT features

# Set base directory
BASE_DIR="/scratch/cl6707/Projects/LLM-AAC/fairseq"
EAT_DIR="${BASE_DIR}/EAT"

# Set the path where you want to save model checkpoints
CHECKPOINT_DIR="/scratch/cl6707/Projects/LLM-AAC/checkpoints/eat_audioset_cwt_precomputed"

# Set the number of GPUs to use
NUM_GPUS=4

# Set the batch size per GPU (adjust based on your GPU memory)
BATCH_SIZE=12

# Set the dataset config (balanced or unbalanced)
DATASET_CONFIG="balanced"

# Set the cache directory and precomputed features path
OUTPUT_DIR="/scratch/cl6707/Shared_Datasets/AudioSet_Wavelets"
PRECOMPUTED_FEATURES="${OUTPUT_DIR}/latest_${DATASET_CONFIG}_train_cwt_features.h5"

# Print configuration
echo "Starting pre-training with pre-computed CWT features..."
echo "Dataset config: ${DATASET_CONFIG}"
echo "Features path: ${PRECOMPUTED_FEATURES}"
echo "Checkpoint directory: ${CHECKPOINT_DIR}"
echo "Number of GPUs: ${NUM_GPUS}"
echo "Batch size per GPU: ${BATCH_SIZE}"

# Check if precomputed features exist, if not generate them
if [ ! -f "${PRECOMPUTED_FEATURES}" ]; then
    echo "Pre-computed features not found at ${PRECOMPUTED_FEATURES}"
    echo "Generating pre-computed features first..."
    
    mkdir -p "${OUTPUT_DIR}"
    
    # Run the precomputation script
    bash "${EAT_DIR}/scripts/precompute_wavelets.sh" \
        --output_dir "${OUTPUT_DIR}" \
        --config "${DATASET_CONFIG}" \
        --split "train"
    
    # Check if precomputation succeeded
    if [ ! -f "${PRECOMPUTED_FEATURES}" ]; then
        echo "Error: Failed to generate precomputed features."
        exit 1
    fi
fi

# Create checkpoint directory if it doesn't exist
mkdir -p "${CHECKPOINT_DIR}"

# Install required dependencies
for pkg in datasets h5py; do
    if ! python -c "import $pkg" &>/dev/null; then
        echo "Installing $pkg..."
        pip install "$pkg"
    fi
done

# Run the pre-training command
cd "${BASE_DIR}"
python fairseq_cli/hydra_train.py -m \
    --config-dir "${EAT_DIR}/config" \
    --config-name pretraining_audioset_hf \
    common.user_dir="${EAT_DIR}" \
    checkpoint.save_dir="${CHECKPOINT_DIR}" \
    distributed_training.distributed_world_size=${NUM_GPUS} \
    dataset.batch_size=${BATCH_SIZE} \
    task.hf_dataset_config="${DATASET_CONFIG}" \
    task.hf_cache_dir="${OUTPUT_DIR}" \
    task.feature_type="cwt" \
    task.use_precomputed_features=true \
    task.precomputed_features_path="${PRECOMPUTED_FEATURES}" \
    task.force_compute=false

echo "Training completed!"