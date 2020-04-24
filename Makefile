osx: src/gecode-support/pybind_gecode.cpp
	rm -rf src/gecode-support/build
	mkdir -p src/gecode-support/build
	cd src/gecode-support/build && \
		cmake .. && \
		make && \
		mv pybind_gecode ../../../.

run:
	docker run -ti -v $(PWD):/app/src dp-cp:latest bash

ppo_example:
	./pybind_gecode -time 30000 -model rl-ppo -seed 15 50

run_example: osx
	./pybind_gecode -time 15000 -model rl-no-cache -seed 15 50

.env: .env.dist
	envsubst < .env.dist > .env

build_gecode: .env
	DOCKER_BUILDKIT=1 docker build -f Dockerfile_gecode -t dp-cp --secret id=env,src=.env .

tag_push_image: build_gecode
	docker tag dp-cp images.borgy.elementai.net/moisan/dp-cp
	docker push images.borgy.elementai.net/moisan/dp-cp

saga: tag_push_image
	saga submit --verbose --config shk.json

saga_minizinc: tag_push_image
	saga submit --verbose --config shk_cp_tsp.json

saga_ppo: tag_push_image
	saga submit --verbose --config shk_ppo.json

saga_ortools: tag_push_image
	saga submit --verbose --config shk_ortools.json

saga_tsptw_ortools: tag_push_image
	saga submit --verbose --config shuriken_configs/shk_tsptw_ortools.json

saga_tsptw_beam: tag_push_image
	saga submit --verbose --config shuriken_configs/shk_tsptw_ortools.json

saga_tsptw: tag_push_image
	saga submit --verbose --config shuriken_configs/shk_tsptw_ortools.json

saga_tsptw_dqn: tag_push_image
	saga submit --verbose --config shuriken_configs/shk_tsptw_dqn.json

saga_tsptw_ppo: tag_push_image
	saga submit --verbose --config shuriken_configs/shk_tsptw_ppo.json

saga_tsptw_luby: tag_push_image
	saga submit --verbose --config shuriken_configs/shk_tsptw_ppo_luby.json

saga_tsptw_nearest: tag_push_image
	saga submit --verbose --config shuriken_configs/shk_tsptw_nearest.json

saga_tsptw_cp: tag_push_image
	saga submit --verbose --config shuriken_configs/shk_tsptw_nearest.json


saga_knapsack: tag_push_image
	saga submit --verbose --config shuriken_configs/shk_knapsack.json

saga_knapsack_luby: tag_push_image
	saga submit --verbose --config shuriken_configs/shk_knapsack_ppo_luby.json

saga_knapsack_optimal: tag_push_image
	saga submit --verbose --config shuriken_configs/shk_knapsack_optimal.json

saga_knapsack_beam: tag_push_image
	saga submit --verbose --config shuriken_configs/shk_knapsack_optimal.json

saga_knapsack_cp: tag_push_image
	saga submit --verbose --config shuriken_configs/shk_knapsack_cp.json

saga_knapsack_cbc: tag_push_image
	saga submit --verbose --config shuriken_configs/shk_knapsack_cp.json

saga_knapsack_beam: tag_push_image
	saga submit --verbose --config shuriken_configs/shk_knapsack_cp.json

linux:
	rm -rf src/gecode-TSP/build
	mkdir src/gecode-TSP/build
	cd src/gecode-TSP/build && \
		cmake .. -DGECODE_ROOT=/home/linuxbrew/.linuxbrew/Cellar/gecode/6.2.0/ \
		-DPYTHON_EXECUTABLE=/opt/conda/envs/dp-cp-gecode/bin/python3 && \
		make && \
		mv TSP_solver ../../../.

linux_tsptw:
	rm -rf src/gecode-TSPTW/build
	mkdir src/gecode-TSPTW/build
	cd src/gecode-TSPTW/build && \
		cmake .. -DGECODE_ROOT=/home/linuxbrew/.linuxbrew/Cellar/gecode/6.2.0/ \
		-DPYTHON_EXECUTABLE=/opt/conda/envs/dp-cp-gecode/bin/python3 && \
		make && \
		mv TSPTW_solver ../../../.

linux_knapsack:
	rm -rf src/gecode-knapsack/build
	mkdir src/gecode-knapsack/build
	cd src/gecode-knapsack/build && \
		cmake .. -DGECODE_ROOT=/home/linuxbrew/.linuxbrew/Cellar/gecode/6.2.0/ \
		-DPYTHON_EXECUTABLE=/opt/conda/envs/dp-cp-gecode/bin/python3 && \
		make && \
		mv knapsack_solver ../../../.

linux_portfolio:
	rm -rf src/gecode-portfolio/build
	mkdir src/gecode-portfolio/build
	cd src/gecode-portfolio/build && \
		cmake .. -DGECODE_ROOT=/home/linuxbrew/.linuxbrew/Cellar/gecode/6.2.0/ \
		-DPYTHON_EXECUTABLE=/opt/conda/envs/dp-cp-gecode/bin/python3 && \
		make && \
		mv portfolio_solver ../../../.

tsptw:
	rm -rf src/problem/tsptw/solving/build
	mkdir src/problem/tsptw/solving/build
	cd src/problem/tsptw/solving/build && \
		cmake .. -DPYTHON_EXECUTABLE:FILEPATH=/Users/quentin/opt/miniconda3/envs/dp-solver-env/bin/python3.7  && \
		make && \
		mv solver_tsptw ../../../../../.
