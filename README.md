# Generic solver for dynamic programs (DP)

[Dynamic Programming](https://en.wikipedia.org/wiki/Dynamic_programming) (DP), which combines both mathematical modeling and computer programming to solve complex optimization problems, has found applications in numerous fields. The well-known challenge one faces with DP is the state-space explosion problem: the number of generated states grows exponentially, which make DP intractable for solving large problems.

In the last years, [Deep Reinforcement Learning](https://arxiv.org/abs/1811.12560) (DRL) has shown its promise for designing good heuristics dedicated to solve 
NP-hard combinatorial optimization problems. However, current approaches have two shortcomings: 
(1) they mainly focus on the standard travelling salesman problem and they cannot be easily extended to other problems, and (2) they only provide an approximate solution with no systematic ways to improve it or to prove optimality.

In another context, [Constraint Programming](https://en.wikipedia.org/wiki/Constraint_programming) (CP) is a generic tool to solve combinatorial optimization problem.
Based on a complete search procedure, it will always find the optimal solution if we allow an execution time large enough. A critical design choice, that makes CP non-trivial to use in practice, is the branching decision, directing how the search space is explored.

In this work, we propose a general and hybrid approach, based on DRL and CP, for solving DPs. In the related paper, we show experimentally show that our solver is efficient to solve two DP problems: the [Travelling Salesman Problem with Time Windows](https://acrogenesis.com/or-tools/documentation/user_manual/manual/tsp/tsptw.html). We show that our framework outperforms the stand-alone RL solution and the standard CP models, while being generic to any DP.

Please be aware that this project is still at research level.

## Content of the repository

This repository contains the implementation of the paper (xxx). For each problem that we have considered, you can find:

* A DP model serving as a basis for the RL environment and the CP model.
*  The RL enviroment and the CP model. 
*  A RL training algorithm based on Deep Q-Learning (DQN).
*  A RL training algorithm based on Proximal Policy Optimization (PPO).
*  The models, and the hyperparameters used, that we trained.
*  Two CP solving algorithms leveraging the learned models: Iterative Limited Discrepancy Search (I-LDS) and Restart Based Search (RBS)
*  A random instance generators for training the model and evaluating the solver.

```bash
.
├── conda_env.yml  # configuration file for the conda environment
├── main_training_x_y.py  # main file for training a model for the problem y using algorithm x
├── run_training_x_y.sh  # script for running the training. It is where you have to enter the parameters 
├── trained_models/  # directory where the models that you train will be saved
├── selected_models/  # models that we used for our experiments
└── src/ 
	├── architecture/ # implementation of the NN used
        ├── util/  #  utilitary code (as the memory replay)
	├── problem/  # problems that we have implemented
        └── tsptw/ 
              ├── environment/ # the generator, and the DP model, acting also as the RL environment
              ├── training/  # PPO and DQN training algorithms
              ├── solving/  # CP model and solving algorithm
        ├── ...      
```
## Installation Instructions

### 1. Importing the repository

```shell
git clone https://github.com/qcappart/dp-solver.git
```
### 2. Setting up the conda virtual environment

```shell
conda env create -f conda_env.yml 
```
### 3. Building Gecode

TODO

## Basic use

### 1. Training a model

```shell
./run_training_ppo_tsptw.sh
```
### 2. Solving the problem

TODO

## Technologies and tools used

* The code, at the exception of the CP model, is implemented in Python 3.6.
* The CP model is implemented in C++ and is solved using [Gecode](https://www.gecode.org/). The reason of this design choice is that there is no CP solver in Python with the requirements we needed. 
* The neural network architecture as been implemented in Pytorch together with DGL. 
* The interface between the C++ and Python code is done with [Pybind11](https://github.com/pybind).

## Current implemented problems

At the moment, only the TSPTW problem is present in this repository. However, we also have the TSP, the 0/1 Knapsack problem, and a nonlinear portfolio optimization problem available, but in non-cleaned code. If there is demand for these problems, I will clean the code and add them in this repository. Feel free to open an issue for that.

## Cite

Please use this reference:

```latex
TODO
```
