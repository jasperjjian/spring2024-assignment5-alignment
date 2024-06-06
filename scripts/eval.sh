#!/bin/bash
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gpus=2
#SBATCH --time=2:00:00
#SBATCH --nodelist=ad12a3ca-02


python3 /home/c-jjian/assignments/spring2024-assignment5-alignment/cs336_alignment/safety_eval.py

python scripts/evaluate_safety.py \
    --input-path '/home/c-jjian/assignments/spring2024-assignment5-alignment/results/simple_safety_tests/llama_3_8b_results.jsonl' \
    --model-name-or-path /home/shared/Meta-Llama-3-70B-Instruct \
    --num-gpus 2 \
    --output-path '/home/c-jjian/assignments/spring2024-assignment5-alignment/results/simple_safety_tests/llama_3_8b.annotated.jsonl' 