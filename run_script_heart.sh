python conceptor_aided_recurrent_autoencoder_robot_human.py \
    --name="heart_norm_comp0.1" \
    --a_dt=0.1 \
    --data="heart_data_40_norm.npy" \
    --mlp_size_hidden="[256,256,256]" \
    --steps_per_eval=500 \
    --learning_rate=0.001 \
    --num_epochs=100000 \
    --mlp_in_out="[[],[]]" \
    --noise=0.0 \
    --p_forcing=False \
    --interp_range=0.005 \
   