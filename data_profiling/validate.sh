#!/bin/bash
#SBATCH --nodes=1                   # 1 node
#SBATCH --ntasks-per-node=1          # 1 task per node
#SBATCH --cpus-per-task=2           
#SBATCH --time=3:00:00
#SBATCH --mem=8GB
#SBATCH --job-name=validate-clips
#SBATCH --output=validate-clips.out
module purge
singularity exec --nv \
    --overlay /scratch/$USER/my_env/overlay-15GB-500K.ext3:rw \
    /scratch/work/public/singularity/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif \
    /bin/bash -c 'source /ext3/env.sh; python validate-clips.py'