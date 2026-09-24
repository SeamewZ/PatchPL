#!/bin/bash
# launch 3-shard evaluation on split_test with vl20-full LoRA
cd /home/wcx/swe
mkdir -p data/swe_verify/eval_vl20_full
for i in 0 1 2; do
  CUDA_VISIBLE_DEVICES=$i EVAL_SHARD=$i EVAL_NUM=3 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  INSTS=data/swe_verify/split_test.jsonl \
  EVAL_OUT=data/swe_verify/eval_vl20_full \
  LORA_PATH=checkpoints/qwen35-4b-lora-vl20-full \
  nohup /home/wcx/miniconda3/bin/python -u eval_vl20.py > logs/eval_test_$i.log 2>&1 &
done
echo LAUNCHED-EVAL-3SHARDS
