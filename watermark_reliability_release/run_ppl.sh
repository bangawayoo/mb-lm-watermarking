#!/bin/bash
export HF_ACCESS_TOKEN="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"
huggingface-cli login --token $HF_ACCESS_TOKEN

export WANDB_API_KEY="2570d8af822487be5bd6478ecc3c153ac9beede5";
wandb offline
export CUDA_VISIBLE_DEVICES="0";

export HF_HOME="/cache"
RUN_NAME="16b-30T-4R"
OUTPUT_DIR="experiments/"

### logging related ###
#OUTPUT_DIR="./test"
WANDB=T
DEBUG=F
##########################

### evaluation related ###
export MSG_LEN=24; export CODE_LEN=24; export RADIX=4; export ZERO_BIT=F;
export TOKEN_LEN=250
export ORACLE_MODEL="meta-llama/Llama-2-13b-hf"
export ORACLE_MODEL="facebook/opt-1.3b"
export IGNORE_R_NGRAM=F
export BATCH_SIZE=4
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
    --message_length="$MSG_LEN" \
    --base=$RADIX \
    --ignore_repeated_ngrams=$IGNORE_R_NGRAM \
    --target_T="$TOKEN_LEN" \
    --debug=$DEBUG