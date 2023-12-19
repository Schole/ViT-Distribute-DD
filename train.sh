#!/bin/bash
#SBATCH -q regular
#SBATCH -N 4
#SBATCH --gpus-per-node 4
#SBATCH -G 16
#SBATCH -C gpu
#SBATCH -t 0:30:0
#SBATCH -J train-vit-4
#SBATCH -o logs/%x-%j.out
#SBATCH -A m4431_g
#SBATCH --mail-user=yw450@rutgers.edu

# Setup software
module load python
module load cudatoolkit/11.7
conda activate cuda2
source /global/homes/y/yw450/.conda/envs/cuda2


export NUMEXPR_MAX_THREADS=64
export KMP_AFFINITY="granularity=fine,compact,1,0"
export KMP_BLOCKTIME=1

export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12802
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`

rm hostfile
for l in `scontrol show hostnames $SLURM_NODELIST`
do
	echo "${l} slots=4" >> hostfile
done

# Tell DeepSpeed where to find our generated hostfile via DLTS_HOSTFILE
export DLTS_HOSTFILE=./hostfiles/hosts_$SLURM_JOBID

# Run the training
srun -l -u torchrun --nproc_per_node=4 train.py -d True --use_cifar_10 True $@ | tee small_vit_16.log

