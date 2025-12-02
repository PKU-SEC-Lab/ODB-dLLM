#!/bin/bash

# Set the environment variables first before running the command.
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

GPU_ID=${1:-1}
TASK=${2:-gsm8k}
MODEL_PATH=${3:-'GSAI-ML/LLaDA-1.5'}
CONFIG_TYPE=${4:-"jump-share"}

echo "Running configuration: $CONFIG_TYPE"

if [ "$CONFIG_TYPE" = "llada_baseline" ]; then
    # LLaDA baseline
    CUDA_VISIBLE_DEVICES=${GPU_ID} python eval_llada_baseline.py \
        --confirm_run_unsafe_code \
        --model llada_single_test \
        --tasks ${TASK} \
        --model_args model_path=${MODEL_PATH},is_check_greedy=False,gen_length=1024,steps=1024,block_length=32

elif [ "$CONFIG_TYPE" = "fast_dllm_baseline" ]; then
    # Fast-dLLM baseline: parallel + dualcache
    CUDA_VISIBLE_DEVICES=${GPU_ID} python eval_fast_dllm.py \
        --confirm_run_unsafe_code \
        --model llada_single_test \
        --tasks ${TASK} \
        --model_args model_path=${MODEL_PATH},task_name=${TASK},gen_length=1024,steps=1024,block_length=32,use_cache=True,dual_cache=True,threshold=0.9,show_speed=True

elif [ "$CONFIG_TYPE" = "alp" ]; then
    # parallel + dualcache + adaptive length prediction
    CUDA_VISIBLE_DEVICES=${GPU_ID} python eval_alp.py \
        --confirm_run_unsafe_code \
        --model llada_single_test \
        --tasks ${TASK} \
        --model_args model_path=${MODEL_PATH},task_name=${TASK},gen_length=1024,steps=1024,block_length=32,use_cache=True,dual_cache=True,threshold=0.9,show_speed=True

elif [ "$CONFIG_TYPE" = "accept-jump" ]; then
    # parallel + dualcache + adaptive length prediction + accept-jump speculative
    CUDA_VISIBLE_DEVICES=${GPU_ID} python eval_alp_accept_jump.py \
        --confirm_run_unsafe_code \
        --model llada_single_test \
        --tasks ${TASK} \
        --model_args model_path=${MODEL_PATH},task_name=${TASK},gen_length=1024,steps=1024,block_length=32,use_cache=True,dual_cache=True,threshold=0.9,show_speed=True

elif [ "$CONFIG_TYPE" = "jump-share" ]; then
    # parallel + dualcache + adaptive length prediction + jump-share speculative
    CUDA_VISIBLE_DEVICES=${GPU_ID} python eval_alp_jump_share.py \
        --confirm_run_unsafe_code \
        --model llada_single_test \
        --tasks ${TASK} \
        --model_args model_path=${MODEL_PATH},task_name=${TASK},gen_length=1024,steps=1024,block_length=32,use_cache=True,dual_cache=True,threshold=0.9,show_speed=True

else
    echo "Unknown configuration type: $CONFIG_TYPE"
    echo "Available configurations: jump-accept, jump-share"
    exit 1
fi
