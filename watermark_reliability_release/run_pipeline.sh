#!/bin/bash
# Script to run the generation, attack, and evaluation steps of the pipeline
# Below variables should be set in a seperate script.

#export HF_DATASETS_CACHE="/cache"
#export HF_HOME="/cache"
#export CUDA_VISIBLE_DEVICES="0"
huggingface-cli login --token $HF_ACCESS_TOKEN



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
#SAMPLING=T
#BS=64 # batch size for generation
##########################

### watermarking related ###
#SEED_SCH="selfhash"
#GAMMA=0.25
#DELTA="2.0"
#MSG_LEN=32 # bit-width
#RADIX=4
#ZERO_BIT=F

## attack related ##
#ATTACK_M=copy-paste
#srcp="80%"
##########################

### logging related ###
#OUTPUT_DIR="./test"
#WANDB=T
##########################

### evaluation related ###
#EVAL_METRICS="repetition"
##########################



### logging ###
#  RUN_NAME="32b"
GENERATION_OUTPUT_DIR="$OUTPUT_DIR"/"$RUN_NAME"
echo "Running generation pipeline with output dir: $GENERATION_OUTPUT_DIR"
###############
if [ $RUN_GEN == T ]
then
  python generation_pipeline.py \
      --model_name=$MODEL_PATH \
      --dataset_name=$D_NAME \
      --dataset_config_name=$D_CONFIG \
      --max_new_tokens=$TOKEN_LEN \
      --min_prompt_tokens=50 \
      --limit_indices=5000 \
      --min_generations=$MIN_GEN \
      --input_truncation_strategy=completion_length \
      --input_filtering_strategy=$INPUT_FILTER \
      --output_filtering_strategy=max_new_tokens \
      --use_sampling $SAMPLING \
      --num_beam $NUM_BEAMS \
      --seeding_scheme=$SEED_SCH \
      --gamma=$GAMMA \
      --delta=$DELTA \
      --base=$RADIX \
      --zero_bit=$ZERO_BIT \
      --use_position_prf=$USE_PPRF \
      --use_fixed_position=$USE_FIXP \
      --use_feedback=$FEEDBACK --feedback_bias=$F_BIAS \
      --feedback_eta=$F_ETA --feedback_tau=$F_TAU \
      --run_name="$RUN_NAME"_gen \
      --wandb=$WANDB \
      --verbose=True \
      --output_dir="$GENERATION_OUTPUT_DIR" \
      --overwrite T \
      --message_length="$MSG_LEN" --code_length="$CODE_LEN" \
      --generation_batch_size="$BS" \
      --load_fp16 $FP16
fi

if [ $RUN_ATT == T ]
then
  python attack_pipeline.py \
      --attack_method="${ATTACK_M}" \
      --run_name="${RUN_NAME}_${ATTACK_M}-${ATTACK_SUFFIX}attack" \
      --wandb=$WANDB \
      --cp_attack_insertion_len "${srcp}" \
      --cp_attack_type=$CP_ATT_TYPE \
      --input_dir="$GENERATION_OUTPUT_DIR" \
      --output_dir="$OUTPUT_DIR"/"${RUN_NAME}_${ATTACK_M}-${ATTACK_SUFFIX}attack" \
      --verbose=True --overwrite_output_file T \
      --limit_rows=$LIMIT_ROWS \
      --order=$DIPPER_ORDER \
      --lex=$DIPPER_LEX

  export GENERATION_OUTPUT_DIR="$OUTPUT_DIR"/"${RUN_NAME}_${ATTACK_M}-${ATTACK_SUFFIX}attack"
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
      --oracle_model_name_or_path $ORACLE_MODEL \
      --evaluation_metrics=$EVAL_METRICS \
      --message_length="$MSG_LEN" \
      --base=$RADIX \
      --ignore_repeated_ngrams=$IGNORE_R_NGRAM \
      --target_T="$TOKEN_LEN" \
      --lower_tolerance_T=$LOWER_TOL --upper_tolerance_T=$UPPER_TOL \
      --debug=$DEBUG --limit_rows=$LIMIT_ROWS
fi
