#!/bin/bash
#SBATCH --time=05:00:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100.20gb:1

module purge
module load Python/3.10.4-GCCcore-11.3.0
module load jax/0.3.25-foss-2022a-CUDA-11.7.0
module load PyTorch/1.12.1-foss-2022a-CUDA-11.7.0
module load TensorFlow/2.11.0-foss-2022a-CUDA-11.7.0

source $HOME/venvs/carae/bin/activate

python3 conceptor_aided_recurrent_autoencoder_robot_human.py \
    --name="human_robot_pp_wout" \
    --a_dt=0.1 \
    --data="robot_human_data_pp.npy" \
    --mlp_size_hidden="[1024,1024,1024]" \
    --steps_per_eval=200 \
    --learning_rate=0.0005 \
    --num_epochs=100000 \
    --mlp_in_out="[[16],[64,64,64]]" \
    --noise=0.05 \
    --p_forcing=False \

deactivate