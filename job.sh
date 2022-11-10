#!/bin/bash
#
#
### comment lines start with ## or #+space
### slurm option lines start with #SBATCH


#SBATCH --job-name=baseline  	## job name
#SBATCH --time=0-72:00:00       ## days-hours:minutes:seconds
#SBATCH --mem=32000M             ##   3GB ram (hardware ratio is < 4GB/core)

### SBATCH --output=$1/train.log	## standard out file
#SBATCH --ntasks=1            ## Ntasks.  default is 1.
#SBATCH --cpus-per-task=1	## Ncores per task.  Use greater than 1 for multi-threaded jobs.  default is 1.
###SBATCH --partition=generic  ##  can specify partition here, but it is pre-empted by what module is loaded
###SBATCH --account=your_tenant_name    ## only need to specify if you belong to multiple tenants on ScienceCluster
#SBATCH --gres gpu:1
#SBATCH --gres gpu:Tesla-V100-32GB:1

module load nvidia/cuda10.2-cudnn7.6.5
module load anaconda3
source activate audio_clf
pip3 install scikit-learn
pip3 install tensorboard
pip3 install torch torchvision torchaudio

CUBLAS_WORKSPACE_CONFIG=:4096:8 stdbuf -o0 -e0 srun --unbuffered python model.py -c $1