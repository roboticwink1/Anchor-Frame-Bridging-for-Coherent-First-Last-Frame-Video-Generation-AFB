#!/bin/bash
#SBATCH -o job.%j_hunyuanvideo_i2v_infer_reverse_all.out
#SBATCH --partition=gpuA800
#SBATCH -J hunyuan_i2v_infer_reverse_all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8


##### Number of total processes
echo "--------------------------------------------------------------------------"
echo "Job ID: $SLURM_JOB_ID"
echo "Allocated nodes:  $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "Total number of tasks: $SLURM_NTASKS"
echo "Number of tasks per node: $SLURM_NTASKS_PER_NODE"
echo "Memory per CPU: $SLURM_MEM_PER_CPU MB"
echo "--------------------------------------------------------------------------"

nvidia-smi
source ~/anaconda3/bin/activate
source ~/.bashrc
conda activate diffsynth-studio

# debugging flags (optional)
#export NCCL_DEBUG=INFO
#export PYTHONFAULTHANDLER=1
#export NCCL_SOCKET_IFNAME=^docker0,lo

cd ~/lmj/kevin/DiffSynth-Studio
python examples/HunyuanVideo/hunyuanvideo_i2v_80G_reverse_all.py