#!/bin/bash
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=train
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=30
#SBATCH --gres=gpu:tesla-smx2:1
#SBATCH --mem-per-cpu=2G
#SBATCH -o test_out.txt
#SBATCH -e test_err.txt

srun python run_with_submitit.py --nodes 8 --ngpus 1  --arch vit_small --epochs 300 --teacher_temp 0.07 --warmup_teacher_temp_epochs 30 --norm_last_layer false --data_path /scratch/itee/uqxxu16/data/imagenet --output_dir /scratch/itee/uqxxu16/dino/output