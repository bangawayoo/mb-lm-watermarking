wandb offline
export OPENAI_API_KEY=""
export WANDB_API_KEY=""
export CUDA_VISIBLE_DEVICES="0";
export HF_HOME="/workspace/cache"
export HF_ACCESS_TOKEN=""
#huggingface-cli login --token $HF_ACCESS_TOKEN
export WANDB=T


### Experiment type ###
export RUN_GEN=T; export RUN_ATT=F; export RUN_EVAL=T; export DEBUG=T;
export LIMIT_ROWS=-1


### Generation ###
export MODEL_PATH="gpt2-xl"; export BS=1; export TOKEN_LEN=100;
export D_NAME="c4"; export D_CONFIG="realnewslike"; export INPUT_FILTER="prompt_and_completion_length"
if [ $D_NAME == "lfqa" ]
then
  export INPUT_FILTER="completion_length"
fi
export NUM_BEAMS=1; export SAMPLING=T
export FP16=T; export MIN_GEN=5
## multi-bit
export MSG_LEN=4; export RADIX=4; export ZERO_BIT=F;
export SEED_SCH="lefthash"; export GAMMA=0.25; export DELTA=2.0;
## logging
export RUN_NAME="${MSG_LEN}b-${TOKEN_LEN}T-${RADIX}R-${SEED_SCH}"
export OUTPUT_DIR="./experiments/sample-run/"


### Attack ###
export ATTACK_M="dipper"; export DIPPER_ORDER=0; export DIPPER_LEX=60;
export ATTACK_M="copy-paste"; export srcp="50%"; export CP_ATT_TYPE="single-single"
export ATTACK_SUFFIX="cp=.5"


### Evaluation ###
export LOWER_TOL=25; export UPPER_TOL=25
export ORACLE_MODEL="meta-llama/Llama-2-13b-hf"
export IGNORE_R_NGRAM=T
export EVAL_METRICS="z-score"


mkdir -p ${OUTPUT_DIR}/log/${RUN_NAME}
bash ./run_pipeline.sh 2>&1 | tee -a ${OUTPUT_DIR}/log/${RUN_NAME}/output.log
cat ${OUTPUT_DIR}/log/${RUN_NAME}/output.log | grep "bit_acc"

