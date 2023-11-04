#!/bin/bash

#export HF_DATASETS_CACHE="/cache"
#export HF_HOME="/cache"
#export CUDA_VISIBLE_DEVICES="0"
huggingface-cli login --token $HF_ACCESS_TOKEN

# Script to run the generation, attack, and evaluation steps of the pipeline
# requires some OUTPUT_DIR to be set in the environment
# as well as a path to the hf format LLAMA model

### experiment types ###
#RUN_GEN=T
#RUN_ATT=T
#RUN_EVAL=T
##########################

### generation related ###
#TOKEN_LEN="500"
#MODEL_PATH="/workspace/Public/llama-2-converted/llama-2-7b-chat/"
#MODEL_PATH="facebook/opt-1.3b"
#MIN_GEN=100 # number of valid samples to generate
SAMPLING=T
#BS=64 # batch size for generation
##########################

### watermarking related ###
#SEED_SCH="selfhash"
#GAMMA=0.25
#DELTA="2.0"
#MSG_LEN=32 # bit-width
#RADIX=4
#ZERO_BIT=F

## attack realted ##
#ATTACK_M=copy-paste
#srcp="80%"
##########################


### logging related ###
#OUTPUT_DIR="./test"
WANDB=T
##########################
GAMMA=("0.125" "0.125" "0.125" "0.25" "0.25" "0.5")
RADIX=(2 4 8 2 4 2)
### evaluation related ###
#EVAL_METRICS="repetition"
##########################

for (( i = 0 ; i < ${#RADIX[@]} ; i++ ))
do
  # attack related
  # SRC_PCT="20% 40% 60% 80%"

  ### logging ###
#  RUN_NAME="32b"
  RUN_NAME="8b-100T-${RADIX[$i]}R-${GAMMA[$i]}gamma"
  GENERATION_OUTPUT_DIR="$OUTPUT_DIR"/"$RUN_NAME"
  echo "Running generation pipeline with output dir: $GENERATION_OUTPUT_DIR"
  ###############

  if [ $RUN_GEN == T ]
  then
    python generation_pipeline.py \
        --model_name=$MODEL_PATH \
        --dataset_name=c4 \
        --dataset_config_name=realnewslike \
        --max_new_tokens=$TOKEN_LEN \
        --min_prompt_tokens=50 \
        --limit_indices=5000 \
        --min_generations=$MIN_GEN \
        --input_truncation_strategy=completion_length \
        --input_filtering_strategy=prompt_and_completion_length \
        --output_filtering_strategy=max_new_tokens \
        --use_sampling $SAMPLING \
        --seeding_scheme=$SEED_SCH \
        --gamma="${GAMMA[$i]}" \
        --delta="${DELTA}" \
        --base="${RADIX[$i]}" \
        --zero_bit=$ZERO_BIT \
        --run_name="$RUN_NAME"_gen \
        --wandb=$WANDB \
        --verbose=True \
        --output_dir="$GENERATION_OUTPUT_DIR" \
        --overwrite T \
        --message_length="$MSG_LEN" \
        --generation_batch_size="$BS" \
        --load_fp16 $FP16
  fi

  if [ $RUN_ATT == T ]
  then
    python attack_pipeline.py \
        --attack_method="${ATTACK_M}" \
        --run_name="${RUN_NAME}_${ATTACK_M}-attack" \
        --wandb=$WANDB \
        --cp_attack_insertion_len "${srcp}" \
        --cp_attack_type triple-single \
        --input_dir="$GENERATION_OUTPUT_DIR" \
        --verbose=True --overwrite_output_file T
  fi

  if [ $RUN_EVAL == T ]
  then
    python evaluation_pipeline.py \
        --evaluation_metrics=all \
        --run_name="$RUN_NAME"_eval \
        --wandb=$WANDB \
        --input_dir="$GENERATION_OUTPUT_DIR" \
        --output_dir="$GENERATION_OUTPUT_DIR"_eval \
        --roc_test_stat=all --overwrite_output_file T --overwrite_args T \
        --evaluation_metrics=$EVAL_METRICS \
        --message_length="$MSG_LEN" \
        --base="${RADIX[$i]}" \
        --target_T="$TOKEN_LEN"
  fi
done
