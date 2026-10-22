cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(pprl CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Torch REQUIRED)
find_package(raylib REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Game Inference / Visualization Binary
add_executable(main main.cpp)
target_link_libraries(main PRIVATE ${TORCH_LIBRARIES} raylib)

# RL Training Binary
add_executable(trainer trainer.cpp)
target_link_libraries(trainer PRIVATE ${TORCH_LIBRARIES} raylib)

# Keep the training loop optimized in Release builds; CMake's default build type
# is unoptimized on many Linux systems.
target_compile_options(trainer PRIVATE $<$<CONFIG:Release>:$<$<CXX_COMPILER_ID:GNU,Clang>:-O3>>)
