#!/bin/bash
# Script for inference with feature-based models with CLAP refinement
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=7

run_dir=/gpfs/scratch/cl6707/Projects/LLM-AAC/SLAM-LLM
cd $run_dir
code_dir=examples/slam_aac

# Choose your model checkpoint - update this path to your trained model
model_dir="/gpfs/scratch/cl6707/Projects/LLM-AAC/exps/Clotho/slam-aac_Clotho_mlp_melspec_wav"

# Define the feature type (mel_spectrogram or cwt)
feature_type="mel_spectrogram"

# Define your inference data
inference_data_path="/gpfs/scratch/cl6707/Shared/clotho/clotho_evaluation_single.jsonl"

# CLAP model path - update to your CLAP model location
clap_dir="/gpfs/scratch/cl6707/Shared/models/clap"

llm_path="lmsys/vicuna-7b-v1.5"
encoder_projector_ds_rate=5

# Create a subdirectory for inference outputs
output_dir="${model_dir}/inference_clap_refined"
mkdir -p $output_dir

# Determine whether we're using a Q-former or MLP (linear) projector
if [[ $model_dir == *"qformer"* ]]; then
    encoder_projector="q-former"
    echo "Using Q-former projector"
else
    encoder_projector="linear"
    echo "Using MLP (linear) projector"
fi

# Determine checkpoint path
if [[ -f "${model_dir}/model.pt" ]]; then
    ckpt_path="${model_dir}/model.pt"
    echo "Found checkpoint at ${ckpt_path}"
else
    # Look for the latest checkpoint
    latest_ckpt=$(find ${model_dir} -name "checkpoint-*.pt" | sort -V | tail -n1)
    if [[ -n "$latest_ckpt" ]]; then
        ckpt_path="$latest_ckpt"
        echo "Using latest checkpoint: ${ckpt_path}"
    else
        echo "No checkpoint found in ${model_dir}. Please check the path."
        exit 1
    fi
fi

# Define the beam size range
beam_range=(2) # 3 4 5 6 7 8)

for num_beams in "${beam_range[@]}"; do
    decode_log="${output_dir}/decode_beam${num_beams}"

    if [ -f "${decode_log}_pred" ]; then
        echo "Decode log ${decode_log}_pred already exists, skipping this beam size..."
        continue
    fi

    echo "Running inference with num_beams=$num_beams"

    # -m debugpy --listen 5678 --wait-for-client
    python $code_dir/inference_aac_batch.py \
        --config-path "conf" \
        --config-name "prompt.yaml" \
        hydra.run.dir=$output_dir \
        ++model_config.llm_name="vicuna-7b-v1.5" \
        ++model_config.llm_path=$llm_path \
        ++model_config.llm_dim=4096 \
        ++model_config.encoder_name=none \
        ++model_config.encoder_path=none \
        ++model_config.encoder_dim=64 \
        ++model_config.encoder_projector=$encoder_projector \
        ++model_config.encoder_projector_ds_rate=$encoder_projector_ds_rate \
        ++dataset_config.encoder_projector_ds_rate=$encoder_projector_ds_rate \
        ++dataset_config.encoder_dim=64 \
        ++dataset_config.dataset=feature_audio_dataset \
        ++dataset_config.file=examples/slam_aac/utils/feature_dataset.py:get_feature_audio_dataset \
        ++dataset_config.data_path=/gpfs/scratch/cl6707/Shared/clotho/clotho_dataset \
        ++dataset_config.feature_type=$feature_type \
        ++dataset_config.val_data_path=$inference_data_path \
        ++dataset_config.inference_mode=true \
        ++dataset_config.fixed_length=true \
        ++dataset_config.target_length=1024 \
        ++dataset_config.prompt="Describe the audio you hear." \
        ++train_config.model_name=aac \
        ++train_config.batching_strategy=custom \
        ++train_config.num_epochs=1 \
        ++train_config.val_batch_size=4 \
        ++train_config.num_workers_dataloader=0 \
        ++train_config.output_dir=$output_dir \
        ++train_config.freeze_encoder=true \
        ++train_config.freeze_llm=true \
        ++train_config.use_peft=false \
        ++ckpt_path=$ckpt_path \
        ++decode_log=$decode_log \
        ++model_config.num_beams=$num_beams \
        ++log_config.log_file="${output_dir}/inference.log"
done

echo "Running CLAP-Refine"

# -m debugpy --listen 6666 --wait-for-client
python ${code_dir}/utils/clap_refine.py \
    --start_beam 2 --end_beam 2 \
    --clap_ckpt $clap_dir/best_model.pt \
    --config $clap_dir/clap_config.yaml \
    --test_jsonl $inference_data_path \
    --exp_explorer $output_dir

echo "CLAP refinement completed. Results saved to $output_dir"

# bash /gpfs/scratch/cl6707/Projects/LLM-AAC/SLAM-LLM/examples/slam_aac/scripts/inference_clotho_CLAP_Refine_feature.sh 