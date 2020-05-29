#!/bin/bash

function timeout() { perl -e 'alarm shift; exec @ARGV' "$@"; }

seed=$1
size=$2
time=$3

time_sec=$((time/1000))

capacity_ratio=0.5
lambda_1=1
lambda_2=5
lambda_3=5
lambda_4=5
discrete_coeff=0
beam_size=16

echo "[DQN]"
timeout $time_sec python src/problem/portfolio/baseline/dqn_solving.py --n_item=$size \
                                                                       --lambda_1=$lambda_1 \
                                                                       --lambda_2=$lambda_2 \
                                                                       --lambda_3=$lambda_3 \
                                                                       --lambda_4=$lambda_4 \
                                                                       --discrete_coeff=$discrete_coeff \
                                                                       --seed=$seed

echo "------------------------------------------------------------------------"

echo "[PPO]"
timeout $time_sec python src/problem/portfolio/baseline/ppo_solving.py --n_item=$size \
                                                                       --lambda_1=$lambda_1 \
                                                                       --lambda_2=$lambda_2 \
                                                                       --lambda_3=$lambda_3 \
                                                                       --lambda_4=$lambda_4 \
                                                                       --discrete_coeff=$discrete_coeff \
                                                                       --seed=$seed \
                                                                       --beam_size=$beam_size


echo "------------------------------------------------------------------------"

echo "[BaB-DQN]"

./solver_portfolio --model=rl-bab-dqn \
                   --time=$time \
                   --size=$size \
                   --capacity_ratio=0.5 \
                   --lambda_1=1 \
                   --lambda_2=5 \
                   --lambda_3=5 \
                   --lambda_4=5 \
                   --discrete_coeffs=0 \
                   --cache=1 \
                   --seed=$seed



echo "------------------------------------------------------------------------"

echo "[ILDS-DQN]"

./solver_portfolio --model=rl-ilds-dqn \
                   --time=$time \
                   --size=$size \
                   --capacity_ratio=0.5 \
                   --lambda_1=1 \
                   --lambda_2=5 \
                   --lambda_3=5 \
                   --lambda_4=5 \
                   --discrete_coeffs=0 \
                   --cache=1 \
                   --d_l=5000 \
                   --seed=$seed


echo "------------------------------------------------------------------------"

echo "[RBS-PPO]"

./solver_portfolio --model=rl-rbs-ppo \
                   --time=$time \
                   --size=$size \
                   --capacity_ratio=0.5 \
                   --lambda_1=1 \
                   --lambda_2=5 \
                   --lambda_3=5 \
                   --lambda_4=5 \
                   --discrete_coeffs=0 \
                   --cache=1 \
                   --luby=1 \
                   --temperature=1 \
                   --seed=$seed




