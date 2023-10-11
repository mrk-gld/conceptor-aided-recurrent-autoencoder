python CARAE_mocap_training.py \
    --name="mocap_conceptor_loss" \
    --num_epochs=1000 \
    --steps_per_eval=50 \
    --rnn_size=500 \
    --learning_rate=5e-3 \
    --beta_1=0.02 \
    --beta_2=0.1 \
    --aperture=10

