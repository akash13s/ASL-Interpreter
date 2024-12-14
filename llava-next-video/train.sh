#!/bin/bash
#SBATCH --nodes=1                   # 1 node
#SBATCH --ntasks-per-node=1          # 1 task per node
#SBATCH --cpus-per-task=1            # 1 CPU per task for 1 GPU
#SBATCH --time=24:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1                 # Request 1 GPU per node
#SBATCH --job-name=llava_next_video
#SBATCH --output=llava_next_video.out
#SBATCH --nice=0

module purge

singularity exec --nv \
    --overlay /scratch/$USER/my_env/overlay-15GB-500K.ext3:rw \
    /scratch/work/public/singularity/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif \
    /bin/bash -c 'source /ext3/env.sh; python ./huggingface_trainer/llava_next_video_trainer.py'