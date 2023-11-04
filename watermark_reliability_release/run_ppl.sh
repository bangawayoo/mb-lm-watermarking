#!/bin/bash
export HF_ACCESS_TOKEN=""
huggingface-cli login --token $HF_ACCESS_TOKEN

export WANDB_API_KEY="";
wandb offline
export CUDA_VISIBLE_DEVICES="0";

export HF_HOME="/workspace/cache"
export RUN_NAME="16b-100T-2R,16b-100T-4R"

export OUTPUT_DIR="experiments/"

### logging related ###
#OUTPUT_DIR="./test"
WANDB=T
DEBUG=F
##########################

### evaluation related ###
export TOKEN_LEN=100
export ORACLE_MODEL="meta-llama/Llama-2-7b-hf"
export IGNORE_R_NGRAM=F
export BATCH_SIZE=1
##########################

GENERATION_OUTPUT_DIR="$OUTPUT_DIR"/"$RUN_NAME"
echo "Running generation pipeline with output dir: $GENERATION_OUTPUT_DIR"

python compute_ppl.py \
    --evaluation_metrics=ppl \
    --run_name="$RUN_NAME" \
    --wandb=$WANDB \
    --input_dir="$OUTPUT_DIR" \
    --roc_test_stat=all --overwrite_output_file T --overwrite_args T \
    --oracle_model_name_or_path $ORACLE_MODEL \
    --ppl_batch_size=${BATCH_SIZE} \
    --target_T="$TOKEN_LEN" \
    --debug=$DEBUG