# Define the range of beta1 and beta2 values
beta1_min=-4
beta1_max=-0.5
beta1_step=0.3

# Create a logscale sequence of values for beta1 and beta2 using awk
beta1_range=($(awk -v min=$beta1_min -v max=$beta1_max -v step=$beta1_step 'BEGIN{for (i=min; i<=max; i+=step) print 10^i}'))
beta2_range=($(awk -v min=$beta1_min -v max=$beta1_max -v step=$beta1_step 'BEGIN{for (i=min; i<=max; i+=step) print 10^i}'))

# Loop over the beta1 and beta2 values
for seed in {1..5}
do
    for beta1 in "${beta1_range[@]}"
    do
        for beta2 in "${beta2_range[@]}"
        do
            # Generate a string that contains the values of beta1 and beta2
            beta_string="beta1_${beta1}_beta2_${beta2}_seed_${seed}"
            echo $beta_string
            # Run the CARAE_sine_wave0.01s_training.py script with the current beta1 and beta2 values as arguments
            python CARAE_mocap_training.py \
            --name=CARAE_mocap_$beta_string \
            --seed=$seed \
            --beta_1=$beta1 \
            --beta_2=$beta2 \
            --plot_interp=False \
            --calc_metric=True \
            --num_epochs=102 \
            --steps_per_eval=20 \
            --rnn_size=512 \
            --learning_rate=1e-2 \
            --aperture=10

            python CARAE_sine_waves_training.py \
            --name=CARAE_sine_$beta_string \
            --seed=$seed \
            --beta_1=$beta1 \
            --beta_2=$beta2 \
            --plot_interp=False \
            --calc_metric=True \
            --num_epochs=202 \
            --steps_per_eval=40 \
            --rnn_size=512 \
            --learning_rate=5e-3 \
            --aperture=10
        done
    done
done