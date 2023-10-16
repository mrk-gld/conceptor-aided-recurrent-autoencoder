python conceptor_aided_recurrent_autoencoder_robot_human.py \
    --name="heart_pp" \
    --a_dt=0.1 \
    --data="heart_data_100.npy" \
    --mlp_size_hidden="[1024,1024,1024]" \
    --steps_per_eval=500 \
    --learning_rate=0.0005 \
    --num_epochs=100000 \
    --mlp_in_out="[[],[]]" \
    --noise=0.05 \
    --p_forcing=False \
   