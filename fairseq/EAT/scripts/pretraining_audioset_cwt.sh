#!/bin/bash

# Set base directory
BASE_DIR="/scratch/cl6707/Projects/LLM-AAC/fairseq"
EAT_DIR="${BASE_DIR}/EAT"

# Set the path where you want to save model checkpoints
CHECKPOINT_DIR="/scratch/cl6707/Projects/LLM-AAC/checkpoints/eat_audioset_cwt"

# Set the number of GPUs to use
NUM_GPUS=4

# Set the batch size per GPU (adjust based on your GPU memory)
BATCH_SIZE=12

# Set the dataset config (balanced or unbalanced)
DATASET_CONFIG="balanced"

# Set the cache directory for HuggingFace datasets
HF_CACHE_DIR="/scratch/cl6707/Shared_Datasets/AudioSet"

# CWT Parameters
CWT_SCALES=64
CWT_WIDTH=8.0

# Check if the required libraries are installed
python -c "import datasets" > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "Installing HuggingFace datasets library..."
    pip install datasets
    
    # Also install required libraries for audio loading
    pip install soundfile librosa
fi

python -c "import pywt" > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "Installing PyWavelets for CWT computation..."
    pip install PyWavelets
fi

python -c "import scipy" > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "Installing SciPy..."
    pip install scipy
fi

echo "Starting pre-training with HuggingFace AudioSet dataset using CWT+MORLET features..."
echo "Dataset config: ${DATASET_CONFIG}"
echo "Dataset cache directory: ${HF_CACHE_DIR}"
echo "Checkpoint directory: ${CHECKPOINT_DIR}"
echo "EAT directory: ${EAT_DIR}"
echo "CWT parameters: scales=${CWT_SCALES}, width=${CWT_WIDTH}"

# Create directories if they don't exist
mkdir -p ${CHECKPOINT_DIR}
mkdir -p ${HF_CACHE_DIR}

# Run the pre-training command
cd ${BASE_DIR}
python fairseq_cli/hydra_train.py -m \
    --config-dir ${EAT_DIR}/config \
    --config-name pretraining_audioset_hf \
    common.user_dir=${EAT_DIR} \
    checkpoint.save_dir=${CHECKPOINT_DIR} \
    distributed_training.distributed_world_size=${NUM_GPUS} \
    dataset.batch_size=${BATCH_SIZE} \
    task.hf_dataset_config=${DATASET_CONFIG} \
    task.hf_cache_dir=${HF_CACHE_DIR} \
    task.feature_type=cwt \
    task.cwt_scales=${CWT_SCALES} \
    task.cwt_width=${CWT_WIDTH} 