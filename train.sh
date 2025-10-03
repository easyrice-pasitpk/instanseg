#!/bin/bash
#SBATCH -p gpu                      # Specify partition [Compute/Memory/GPU]
#SBATCH -N 1 -c 32   			    # Specify number of nodes and processors per task
#SBATCH --mem=128G
#SBATCH --ntasks-per-node=1		    # Specify tasks per node
#SBATCH --gpus=1	                # Specify total number of GPUs
#SBATCH -t 1-00:00:00                # Specify maximum time limit (hour: minute: second)
#SBATCH -A lt200377               	# Specify project name
#SBATCH -J instanseg               	# Specify job name

source .venv/bin/activate
python instanseg/scripts/train.py \
    --data_path /project/lt200377-mpind/segment/ \
    --dataset ez20251002 \
    --model_path /project/lt200377-mpind/segment/models/instanseg \
    --output_path /project/lt200377-mpind/segment/models/instanseg \
    --experiment_str ez20251002_instseg \
    --device cuda:0 \
    --num_workers 8 \
    --fp16 True \
    --batch_size 16 \
    --accumulation_steps 8 \
    --num_epochs 1000 \
    --lr 0.001 \
    --optimizer adamw \
    --tile_size 512 \
    --window_size 256 \
    --augmentation_type minimal \
    --freeze_main_model False \
    -anneal True \
    --rng_seed 5555
