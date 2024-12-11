#!/bin/bash
#SBATCH --nodes=1                   # 1 node
#SBATCH --ntasks-per-node=1          # 1 task per node
#SBATCH --cpus-per-task=2            # 2 CPUs per task for 2 GPUs
#SBATCH --time=24:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:2                 # Request 2 GPUs per node
#SBATCH --job-name=video-llava-2-gpu
#SBATCH --output=video-llava-2-gpu.out
#SBATCH --nice=0

module purge

singularity exec --nv \
    --overlay /scratch/$USER/my_env/overlay-15GB-500K.ext3:rw \
    /scratch/work/public/singularity/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif \
    /bin/bash -c 'source /ext3/env.sh; python train_video_llava.py'

