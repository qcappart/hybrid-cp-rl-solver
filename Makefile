
PYTHON_PATH=~/opt/miniconda3/envs/dp-solver-env/bin/python3.7

tsptw:
	rm -rf src/problem/tsptw/solving/build
	mkdir src/problem/tsptw/solving/build
	cd src/problem/tsptw/solving/build && \
		cmake .. -DPYTHON_EXECUTABLE:FILEPATH=$(PYTHON_PATH)  && \
		make && \
		mv solver_tsptw ../../../../../.

portfolio:
	rm -rf src/problem/portfolio/solving/build
	mkdir src/problem/portfolio/solving/build
	cd src/problem/portfolio/solving/build && \
		cmake .. -DPYTHON_EXECUTABLE:FILEPATH=$(PYTHON_PATH)  && \
		make && \
		mv solver_portfolio ../../../../../.

