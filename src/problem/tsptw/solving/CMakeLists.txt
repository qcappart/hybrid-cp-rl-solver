cmake_minimum_required(VERSION 3.0)
project(solver)

find_package(pybind11 REQUIRED)  # or `add_subdirectory(pybind11)`

SET(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_MODULE_PATH})
find_package(Gecode 6.2 COMPONENTS Driver Float Int Kernel Minimodel Search Set Support)
find_package(Threads)

add_executable(solver_tsptw solver.cpp)
# Link Gecode to target test
target_link_libraries(solver_tsptw PRIVATE pybind11::embed
    Gecode::Driver
    Gecode::Minimodel
    Gecode::Int
    Gecode::Float
    Gecode::Set
    Gecode::Kernel
    Gecode::Minimodel
    Gecode::Search
    Gecode::Support
    Threads::Threads)
