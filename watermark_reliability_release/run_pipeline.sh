CUDA_VISIBLE_DEVICES=1
# Script to run the generation, attack, and evaluation steps of the pipeline
export HF_HOME=$HF_DATASETS_CACHE
# requires some OUTPUT_DIR to be set in the environment
# as well as a path to the hf format LLAMA model

RUN_NAME=tmp
OUTPUT_DIR=test
GENERATION_OUTPUT_DIR="$OUTPUT_DIR"/"$RUN_NAME"
WANDB=T

MODEL_PATH="facebook/opt-1.3b"

MIN_GEN=100

echo "Running generation pipeline with output dir: $GENERATION_OUTPUT_DIR"

#python generation_pipeline.py \
#    --model_name=$MODEL_PATH \
#    --dataset_name=c4 \
#    --dataset_config_name=realnewslike \
#    --max_new_tokens=200 \
#    --min_prompt_tokens=50 \
#    --min_generations=$MIN_GEN \
#    --input_truncation_strategy=completion_length \
#    --input_filtering_strategy=prompt_and_completion_length \
#    --output_filtering_strategy=max_new_tokens \
#    --seeding_scheme=selfhash \
#    --gamma=0.25 \
#    --delta=2.0 \
#    --run_name="$RUN_NAME"_gen \
#    --wandb=$WANDB \
#    --verbose=True \
#    --output_dir=$GENERATION_OUTPUT_DIR


ATTACK_M=copy-paste

python attack_pipeline.py \
    --attack_method=$ATTACK_M \
    --run_name="$RUN_NAME"_gpt_attack \
    --wandb=$WANDB \
    --input_dir=$GENERATION_OUTPUT_DIR \
    --verbose=True --overwrite_output_file T

python evaluation_pipeline.py \
    --evaluation_metrics=all \
    --run_name="$RUN_NAME"_eval \
    --wandb=$WANDB \
    --input_dir=$GENERATION_OUTPUT_DIR \
    --output_dir="$GENERATION_OUTPUT_DIR"_eval \
    --roc_test_stat=all --overwrite_output_file T \
    --evaluation_metrics "z-score"
#    --evaluation_metrics "z-score"