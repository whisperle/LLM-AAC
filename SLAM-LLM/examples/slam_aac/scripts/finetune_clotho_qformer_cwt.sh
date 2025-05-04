#!/bin/bash
# Completely disable bitsandbytes for HPC compatibility
unset BNB_CUDA_VERSION
export USE_8BIT_ADAM=0
export PYTORCH_NO_CUDA_MEMORY_CACHING=1

export PYTHONPATH=/gpfs/scratch/cl6707/Projects/LLM-AAC/fairseq:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=7

# Install PyWavelets for CWT feature extraction if not already installed
python -c "import pywt" || pip install --user PyWavelets

run_dir=/gpfs/scratch/cl6707/Projects/LLM-AAC/SLAM-LLM
cd $run_dir
code_dir=examples/slam_aac

encoder_fairseq_dir=/gpfs/scratch/cl6707/Projects/LLM-AAC/fairseq/EAT

llm_path=lmsys/vicuna-7b-v1.5

seed=10086
btz=4
lr=1e-4  # Higher learning rate for Q-former only training
encoder_projector_ds_rate=5

# Path to the Clotho dataset directory
clotho_dir=/gpfs/scratch/cl6707/Shared/clotho/clotho_dataset
feature_type=cwt  # Using continuous wavelet transform features directly
cwt_wavelet=morl  # Correct wavelet name for PyWavelets (was 'morlet')

exp_name=slam-aac_Clotho_qformer_cwt_wav
output_dir=/gpfs/scratch/cl6707/Projects/LLM-AAC/exps/Clotho/${exp_name}

# Define a simple prompt without spaces
prompt="Describe_the_audio"

hydra_args="
hydra.run.dir=$output_dir \
++model_config.llm_name=vicuna-7b-v1.5 \
++model_config.llm_path=lmsys/vicuna-7b-v1.5 \
++model_config.llm_dim=4096 \
++model_config.encoder_name=none \
++model_config.encoder_ds_rate=2 \
++model_config.encoder_projector_ds_rate=$encoder_projector_ds_rate \
++model_config.encoder_path=none \
++model_config.encoder_dim=64 \
++model_config.encoder_projector=q-former \
++model_config.encoder_projector_num_queries=32 \
++model_config.encoder_projector_num_layers=2 \
++model_config.encoder_projector_hidden_dim=768 \
++model_config.encoder_projector_num_heads=12 \
++model_config.qformer_layers=2 \
++model_config.query_tokens=32 \
++model_config.qformer_hidden_dim=768 \
++model_config.qformer_heads=12 \
++model_config.encoder_fairseq_dir=$encoder_fairseq_dir \
++dataset_config.encoder_projector_ds_rate=${encoder_projector_ds_rate} \
++dataset_config.encoder_dim=64 \
++dataset_config.dataset=feature_audio_dataset \
++dataset_config.file=examples/slam_aac/utils/feature_dataset.py:get_feature_audio_dataset \
++dataset_config.data_path=$clotho_dir \
++dataset_config.feature_type=$feature_type \
++dataset_config.cwt_wavelet=$cwt_wavelet \
++dataset_config.fixed_length=true \
++dataset_config.fix_length_audio=480000 \
++dataset_config.target_length=1024 \
++dataset_config.mel_size=64 \
++dataset_config.prompt=$prompt \
++train_config.model_name=aac \
++train_config.freeze_llm=true \
++train_config.batching_strategy=custom \
++train_config.warmup_steps=500 \
++train_config.total_steps=20000 \
++train_config.lr=$lr \
++train_config.validation_interval=500 \
++train_config.batch_size_training=$btz \
++train_config.val_batch_size=$btz \
++train_config.num_workers_dataloader=4 \
++train_config.use_fp16=false \
++train_config.output_dir=$output_dir \
++train_config.seed=${seed} \
++train_config.use_peft=false \
++train_config.enable_fsdp=false \
++train_config.enable_ddp=false \
++train_config.one_gpu=true \
++train_config.quantization=false \
++log_config.log_file=\"${output_dir}/train.log\" \
++log_config.wandb_dir=${output_dir} \
++log_config.wandb_entity_name=cl6707 \
++log_config.wandb_project_name=slam-llm \
++log_config.wandb_exp_name=$exp_name \
++log_config.use_wandb=true \
++metric=acc \
"

# Run the training
python $code_dir/finetune_aac.py \
    --config-path "conf" \
    --config-name "prompt.yaml" \
    $hydra_args

# bash /gpfs/scratch/cl6707/Projects/LLM-AAC/SLAM-LLM/examples/slam_aac/scripts/finetune_clotho_qformer_cwt.sh 