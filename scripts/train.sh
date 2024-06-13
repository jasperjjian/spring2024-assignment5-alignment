#!/bin/bash
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --nodelist=ad12a3ca-02


python3 /home/c-jjian/assignments/spring2024-assignment5-alignment/cs336_alignment/train_sft.py
