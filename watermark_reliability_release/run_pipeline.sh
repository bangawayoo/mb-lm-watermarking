export CUDA_VISIBLE_DEVICES="1"
wandb offline

# Script to run the generation, attack, and evaluation steps of the pipeline
export HF_HOME=$HF_DATASETS_CACHE
# requires some OUTPUT_DIR to be set in the environment
# as well as a path to the hf format LLAMA model

RUN_NAME=tmp
OUTPUT_DIR=test_sb
WANDB=T

MODEL_PATH="facebook/opt-1.3b"
MODEL_PATH="gpt2"

MIN_GEN=10
SEED_SCH="selfhash"
GAMMA=0.25
DELTA=2.0
SAMPLING=True

GENERATION_OUTPUT_DIR="$OUTPUT_DIR"/"$RUN_NAME"
echo "Running generation pipeline with output dir: $GENERATION_OUTPUT_DIR"

RUN_GEN=T
if [ $RUN_GEN == T ]; then
  python generation_pipeline.py \
      --model_name=$MODEL_PATH \
      --dataset_name=c4 \
      --dataset_config_name=realnewslike \
      --max_new_tokens=500 \
      --min_prompt_tokens=50 \
      --limit_indices 1000 \
      --min_generations=$MIN_GEN \
      --input_truncation_strategy=completion_length \
      --input_filtering_strategy=prompt_and_completion_length \
      --output_filtering_strategy=max_new_tokens \
      --use_sampling $SAMPLING \
      --seeding_scheme=$SEED_SCH \
      --gamma=$GAMMA \
      --delta=$DELTA \
      --run_name="$RUN_NAME"_gen \
      --wandb=$WANDB \
      --verbose=True \
      --output_dir=$GENERATION_OUTPUT_DIR \
      --overwrite T
fi


ATTACK_M=copy-paste

RUN_ATT=T
if [ $RUN_ATT == T ]; then
  python attack_pipeline.py \
      --attack_method=$ATTACK_M \
      --run_name="$RUN_NAME"_gpt_attack \
      --wandb=$WANDB \
      --input_dir=$GENERATION_OUTPUT_DIR \
      --verbose=True --overwrite_output_file T
fi


RUN_EVAL=T
if [ $RUN_EVAL == T ]; then
  python evaluation_pipeline.py \
      --evaluation_metrics=all \
      --run_name="$RUN_NAME"_eval \
      --wandb=$WANDB \
      --input_dir=$GENERATION_OUTPUT_DIR \
      --output_dir="$GENERATION_OUTPUT_DIR"_eval \
      --roc_test_stat=all --overwrite_output_file T --overwrite_args T \
      --evaluation_metrics "z-score"
fi
