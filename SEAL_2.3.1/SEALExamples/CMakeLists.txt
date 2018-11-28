cmake_minimum_required(VERSION 3.10)

project(SEALExamples VERSION 2.3.1 LANGUAGES CXX)

# Executable will be in ../bin
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/../bin)

add_executable(sealexamples main.cpp)

# Import SEAL
find_package(SEAL 2.3.1 EXACT REQUIRED)

# Link SEAL
target_link_libraries(sealexamples SEAL::seal)
