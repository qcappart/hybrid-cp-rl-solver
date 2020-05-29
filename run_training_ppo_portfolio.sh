#!/bin/bash

# Seed for the random generation: ensure that the validation set remains the same.
seed=1

# Characterics of the training instances
n_item=20
capacity_ratio=0.5
lambda_1=1
lambda_2=5
lambda_3=5
lambda_4=5

# Parameters for the training
k_epochs=4
update_timestep=2048
learning_rate=0.0001
entropy_value=0.001
eps_clip=0.1
batch_size=128 # batch size must be a divisor of update_timestep
latent_dim=128
hidden_layer=2

# Others
plot_training=1 # Boolean value: plot the training curve or not
mode=cpu

# Folder to save the trained model
network_arch=hidden_layers-$hidden_layer-latent_dim-$latent_dim/
result_root=trained-models/ppo/portfolio/n-item-$n_item/capacity_ratio-$capacity_ratio/lambdas-$lambda_1-$lambda_2-$lambda_3-$lambda_4/seed-$seed/$network_arch
save_dir=$result_root/k_epochs-$k_epochs-update_timestep-$update_timestep-batch_size-$batch_size-learning_rate-$learning_rate-entropy_value-$entropy_value-eps_clip-$eps_clip


if [ ! -e $save_dir ];
then
    mkdir -p $save_dir
fi

python src/problem/portfolio/main_training_ppo_portfolio.py \
    --seed $seed  \
    --n_item $n_item  \
    --capacity_ratio $capacity_ratio \
    --lambda_1 $lambda_1 \
    --lambda_2 $lambda_2 \
    --lambda_3 $lambda_3 \
    --lambda_4 $lambda_4 \
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


