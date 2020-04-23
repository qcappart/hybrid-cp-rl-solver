#!/bin/bash

# Seed for the random generation: ensure that the validation set remains the same.
seed=1

# Characterics of the training instances
n_city=10

# Parameters for the training
k_epochs=4
update_timestep=64
learning_rate=0.0001
entropy_value=0.001
eps_clip=0.1
batch_size=32 # batch size must be a divisor of update_timestep
latent_dim=64
hidden_layer=3

# Others
plot_training=1 # Boolean value: plot the training curve or not
mode=cpu

# Folder to save the trained model
network_arch=hidden_layers-$hidden_layer-latent_dim-$latent_dim/
result_root=trained_models/ppo/tsptw/n_city-$n_city/seed-$seed/$network_arch
save_dir=$result_root/k_epochs-$k_epochs-update_timestep-$update_timestep-batch_size-$batch_size-learning_rate-$learning_rate-entropy_value-$entropy_value-eps_clip-$eps_clip


if [ ! -e $save_dir ];
then
    mkdir -p $save_dir
fi

python main_training_ppo_tsptw.py \
    --seed $seed  \
    --n_city $n_city  \
    --k_epochs $k_epochs \
    --update_timestep $update_timestep \
    --learning_rate $learning_rate \
    --eps_clip $eps_clip \
    --entropy_value $entropy_value \
    --batch_size $batch_size \
    --latent_dim $latent_dim  \
    --hidden_layer $hidden_layer  \
    --save_dir $save_dir  \
    --plot_training $plot_training  \
    --mode $mode \
    2>&1 | tee $save_dir/log-training.txt


