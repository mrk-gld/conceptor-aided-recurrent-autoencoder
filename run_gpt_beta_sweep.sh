# values=(0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 0.5)
values=(0.0001 0.001 0.01 0.1)

# Iterate over the values
for beta1 in "${values[@]}"; do
    # Replace the following line with your command, using $value as needed
    for beta2 in "${values[@]}"; do
        echo "Running command with beta1=$beta1, beta2=$beta2"
        beta1str="${beta1#0.}"
        beta2str="${beta2#0.}"
        python CARAE_sine_waves_training_gpt.py --name=gpt_CARAE_$beta1str_$beta2str --num_epochs=10000 --n_layers=3 --n_heads=4 --conceptor_loss --steps_per_eval=200 --conceptor_layers=1,2 --len_cueing=20 --beta_1=$beta1 --beta_2=$beta2
    done
done
