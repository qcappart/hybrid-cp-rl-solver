# Hybrid solving process for combinatorial optimization problems

Combinatorial optimization has found applications in numerous fields, from aerospace to transportation planning and economics.
The goal is to find an optimal solution among a finite set of possibilities. The well-known challenge one faces with combinatorial optimization is the state-space explosion problem: 
the number of possibilities grows exponentially with the problem size, which makes solving intractable for large problems.

In the last years, [Deep Reinforcement Learning](https://arxiv.org/abs/1811.12560) (DRL) has shown its promise for designing good heuristics dedicated to solve 
NP-hard combinatorial optimization problems. However, current approaches have two shortcomings: 
(1) they mainly focus on the standard travelling salesman problem and they cannot be easily extended to other problems, and (2) they only provide an approximate solution with no systematic ways to improve it or to prove optimality.

In another context, [Constraint Programming](https://en.wikipedia.org/wiki/Constraint_programming) (CP) is a generic tool to solve combinatorial optimization problems.
Based on a complete search procedure, it will always find the optimal solution if we allow an execution time large enough. A critical design choice, that makes CP non-trivial to use in practice, is the branching decision, directing how the search space is explored.
In this work, we propose a general and hybrid approach, based on DRL and CP, for solving combinatorial optimization problems. The core of our approach is based on a [Dynamic Programming](https://en.wikipedia.org/wiki/Dynamic_programming) (DP) formulation, that acts as a bridge between both techniques.

In this work, we propose a general and hybrid approach, based on DRL and CP, for solving combinatorial optimization problems formulated as a DP. In the related paper, we show experimentally show that our solver is efficient to solve two challenging problems: the [Travelling Salesman Problem with Time Windows](https://acrogenesis.com/or-tools/documentation/user_manual/manual/tsp/tsptw.html)
and the [4-moments Portfolio Optimization Problem](https://en.wikipedia.org/wiki/Portfolio_optimization), that includes the *means*, *deviations*, *skewnessess*, and *kurtosis* of the assets. 
Results obtained show that the framework introduced outperforms the stand-alone RL and CP solutions, while being competitive with industrial solvers.

Please be aware that this project is still at research level.

## Content of the repository

For each problem that we have considered, you can find:

* A DP model serving as a basis for the RL environment and the CP model.
*  The RL enviroment and the CP model. 
*  A RL training algorithm based on Deep Q-Learning (DQN).
*  A RL training algorithm based on Proximal Policy Optimization (PPO).
*  The models, and the hyperparameters used, that we trained.
*  Three CP solving algorithms leveraging the learned models: Depth-First Branch-and_bound (BaB), Iterative Limited Discrepancy Search (ILDS), and Restart Based Search (RBS)
*  A random instance generators for training the model and evaluating the solver.

```bash
.
├── conda_env.yml  # configuration file for the conda environment
├── run_training_x_y.sh  # script for running the training. It is where you have to enter the parameters 
├── trained_models/  # directory where the models that you train will be saved
├── selected_models/  # models that we used for our experiments
└── src/ 
	├── architecture/ # implementation of the NN used
        ├── util/  #  utilitary code (as the memory replay)
	├── problem/  # problems that we have implemented
		└── tsptw/ 
		      ├── main_training_x_y.py  # main file for training a model for the problem y using algorithm x
		      ├── baseline/ # methods that are used for comparison
		      ├── environment/ # the generator, and the DP model, acting also as the RL environment
		      ├── training/  # PPO and DQN training algorithms
		      ├── solving/  # CP model and solving algorithm
		├── portfolio/    
```
## Installation instructions

### 1. Importing the repository

```shell
git clone https://github.com/qcappart/hybrid-cp-rl-solver.git
```
### 2. Setting up the conda virtual environment

```shell
conda env create -f conda_env.yml 
```
Note: install a [DGL version](https://www.dgl.ai/pages/start.html) compatible with your CUDA installation.
### 3. Building Gecode

Please refer to the setup instructions available on the [official website](https://www.gecode.org/).

### 4. Compiling the solver

A makefile is available in the root repository. First, modify it by adding your python path. Then, you can compile the project as follows:

```shell
make [problem] # e.g. make tsptw
```
It will create the executable ```solver_tsptw```.

## Basic use

### 1. Training a model
(Does not require Gecode)
```shell
./run_training_ppo_tsptw.sh # for PPO
./run_training_dqn_tsptw.sh # for DQN
```
### 2. Solving the problem
(Require Gecode)
```shell
# For TSPTW
./solver_tsptw --model=rl-ilds-dqn --time=60000 --size=20 --grid_size=100 --max_tw_size=100 --max_tw_gap=10 --d_l=5000 --cache=1 --seed=1  # Solve with ILDS-DQN
./solver_tsptw --model=rl-bab-dqn --time=60000 --size=20 --grid_size=100 --max_tw_size=100 --max_tw_gap=10 --cache=1 --seed=1 # Solve with BaB-DQN
./solver_tsptw --model=rl-rbs-ppo --time=60000 --size=20 --grid_size=100 --max_tw_size=100 --max_tw_gap=10 --cache=1 --luby=1 --temperature=1 --seed=1 # Solve with RBS-PPO
./solver_tsptw --model=nearest --time=60000 --size=20 --grid_size=100 --max_tw_size=100 --max_tw_gap=10 --d_l=5000 --seed=1 # Solve with a nearest neigbour heuristic (no learning)

# For Portfolio
./solver_portfolio --model=rl-ilds-dqn --time=60000 --size=50 --capacity_ratio=0.5 --lambda_1=1 --lambda_2=5 --lambda_3=5 --lambda_4=5  --discrete_coeffs=0 --cache=1 --seed=1 

```
For learning based methods, the model selected by default is the one located in the corresponding ```selected_model/``` repository. For instance:

```shell
selected-models/ppo/tsptw/n-city-20/grid-100-tw-10-100/ 

```

## Example of results

The table recaps the solution obtained for an instance generated with a seed of 0, and a timeout of 60 seconds. 
Bold results indicate that the solver has been able to proof the optimality of the solution and a dash that no solution has been
found within the time limit.

### Tour cost for the TSPTW

| Model name  | 20 cities | 50 cities | 100 cities |
| ------------------ 	|---------------- 	| -------------- 	| --------------|
| DQN  			|    959        	|     -     		|      -       	| 
| PPO (beam-width=16)   |    959        	|     -    		|      -       	| 
| CP-nearest  		|    **959**        	|     -     		|      -       	| 
| BaB-DQN   		|     **959**       	|      **2432**        	|     4735     	| 
| ILDS-DQN   		|    **959**           	|      **2432**      	|     -      	| 
| RBS-PPO   		|    **959**          	|      **2432**     	|      4797     | 

```shell
./benchmarking/tsptw_bmk.sh 0 20 60000 # Arguments: [seed] [n_city] [timeout - ms]
./benchmarking/tsptw_bmk.sh 0 50 60000
./benchmarking/tsptw_bmk.sh 0 100 60000
```

### Profit for Portfolio Optimization

| Model name  		| 20 items 	    | 50 items       	| 100 items      |
| ------------------ 	|----------------   | -------------- 	| -------------- |
| DQN  	  		|     247.40        |      1176.94     |      2223.09      | 
| PPO (beam-width=16)  	|     264.49        |      1257.42      |      2242.67      | 
| BaB-DQN   		|     **273.04**    |      1228.03      |      2224.44      | 
| ILDS-DQN   		|     273.04        |      1201.53      |      2235.89       | 
| RBS-PPO   		|     267.05       |      1265.50      |      2258.65       | 

```shell
./benchmarking/portfolio_bmk.sh 0 20 60000 # Arguments: [seed] [n_item] [timeout - ms]
./benchmarking/portfolio_bmk.sh 0 50 60000
./benchmarking/portfolio_bmk.sh 0 100 60000
```

## Technologies and tools used

* The code, at the exception of the CP model, is implemented in Python 3.7.
* The CP model is implemented in C++ and is solved using [Gecode](https://www.gecode.org/). The reason of this design choice is that there is no CP solver in Python with the requirements we needed. 
* The graph neural network architecture has been implemented in Pytorch together with DGL. 
* The set embedding is based on [SetTransformer](https://github.com/juho-lee/set_transformer).
* The interface between the C++ and Python code is done with [Pybind11](https://github.com/pybind).

## Current implemented problems

At the moment, only the travelling salesman problem with time windows and the 4-moments portfolio optimization are present in this repository. However, we also have the TSP, and the 0-1 Knapsack problem available. If there is demand for these problems, I will add them in this repository. Feel free to open an issue for that or if you want to add another problem.

## Cite

Please use this reference:

```latex
@misc{cappart2020combining,
    title={Combining Reinforcement Learning and Constraint Programming for Combinatorial Optimization},
    author={Quentin Cappart and Thierry Moisan and Louis-Martin Rousseau and Isabeau Prémont-Schwarz and Andre Cire},
    year={2020},
    eprint={2006.01610},
    archivePrefix={arXiv},
    primaryClass={cs.AI}
}
```

## Licence

This work is under MIT licence (https://choosealicense.com/licenses/mit/). It is a short and simple very permissive license with conditions only requiring preservation of copyright and license notices. Licensed works, modifications, and larger works may be distributed under different terms and without source code. 
