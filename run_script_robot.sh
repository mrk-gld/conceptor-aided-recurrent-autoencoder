python conceptor_aided_recurrent_autoencoder_robot_human.py \
    --name="human_robot_pp_wout" \
    --a_dt=0.1 \
    --data="robot_human_data_pp.npy" \
    --mlp_size_hidden="[512,512,512]" \
    --steps_per_eval=100 \
    --learning_rate=0.001 \
    --num_epochs=100000 \
    --mlp_in_out="[[16],[64,64,64]]" \
    --noise=0.0 \
    --p_forcing=True \
    --loading="./logs/human_robot_pp_wout_4/ckpt/params_35001.npz"
   