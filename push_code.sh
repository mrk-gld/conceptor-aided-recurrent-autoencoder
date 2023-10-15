rsync -avz --include 'rnn_utils.py' \
    --include 'conceptor_aided_recurrent_autoencoder_robot_human.py' \
    --include 'robot_data/' \
    --exclude '*' \
    . p302922@login2.hb.hpc.rug.nl:test/