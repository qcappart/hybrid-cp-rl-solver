
PYTHON_PATH="$(CONDA_PREFIX)/bin/python3"

tsptw:
	rm -rf src/problem/tsptw/solving/build
	cmake -Hsrc/problem/tsptw/solving -Bsrc/problem/tsptw/solving/build -DPYTHON_EXECUTABLE:FILEPATH=$(PYTHON_PATH) -DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=TRUE -DCMAKE_BUILD_TYPE:STRING="Release" -G "Unix Makefiles" -DPYTHON_EXECUTABLE:FILEPATH=$(PYTHON_PATH)
	cmake --build src/problem/tsptw/solving/build --config Release --target all -- -j $(nproc) VERBOSE=1
	mv src/problem/tsptw/solving/build/solver_tsptw ./

portfolio:
	rm -rf src/problem/portfolio/solving/build
	cmake -Hsrc/problem/portfolio/solving -Bsrc/problem/portfolio/solving/build -DPYTHON_EXECUTABLE:FILEPATH=$(PYTHON_PATH) -DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=TRUE -DCMAKE_BUILD_TYPE:STRING="Release" -G "Unix Makefiles" -DPYTHON_EXECUTABLE:FILEPATH=$(PYTHON_PATH)
	cmake --build src/problem/portfolio/solving/build --config Release --target all -- -j $(nproc) VERBOSE=1
	mv src/problem/portfolio/solving/build/solver_portfolio ./

