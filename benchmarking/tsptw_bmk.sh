#!/bin/bash

function timeout() { perl -e 'alarm shift; exec @ARGV' "$@"; }

seed=$1
size=$2
time=$3
time_sec=$((time/1000))
grid_size=100
max_tw_size=100
max_tw_gap=10
beam_size=16

echo "[DQN]"
timeout $time_sec python src/problem/tsptw/baseline/dqn_solving.py --n_city=$size \
                                                 --grid_size=$grid_size \
                                                 --max_tw_size=$max_tw_size \
                                                 --max_tw_gap=$max_tw_gap \
                                                 --seed=$seed
echo "------------------------------------------------------------------------"

echo "[PPO]"
timeout $time_sec python src/problem/tsptw/baseline/ppo_solving.py --n_city=$size \
                                                 --grid_size=$grid_size \
                                                 --max_tw_size=$max_tw_size \
                                                 --max_tw_gap=$max_tw_gap \
                                                 --seed=$seed \
                                                 --beam_size=$beam_size
echo "------------------------------------------------------------------------"

echo "[CP-nearest]"
./solver_tsptw --model=nearest \
               --time=$time \
               --size=$size \
               --grid_size=$grid_size \
               --max_tw_size=$max_tw_size \
               --max_tw_gap=$max_tw_gap \
               --d_l=5000 \
               --cache=1 \
               --seed=$seed
echo "------------------------------------------------------------------------"
echo "[BaB-DQN]"
./solver_tsptw --model=rl-bab-dqn \
               --time=$time \
               --size=$size \
               --grid_size=$grid_size \
               --max_tw_size=$max_tw_size \
               --max_tw_gap=$max_tw_gap \
               --cache=1 \
               --seed=$seed
echo "------------------------------------------------------------------------"
echo "[ILDS-DQN]"
./solver_tsptw --model=rl-ilds-dqn \
               --time=$time \
               --size=$size \
               --grid_size=$grid_size \
               --max_tw_size=$max_tw_size \
               --max_tw_gap=$max_tw_gap \
               --d_l=5000 \
               --cache=1 \
               --seed=$seed
echo "------------------------------------------------------------------------"
echo "[RBS-PPO]"
./solver_tsptw --model=rl-rbs-ppo \
               --time=$time \
               --size=$size \
               --grid_size=$grid_size \
               --max_tw_size=$max_tw_size \
               --max_tw_gap=$max_tw_gap \
               --luby=128 \
               --temperature=20 \
               --cache=1 \
               --seed=$seed


