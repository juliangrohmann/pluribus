cmake_minimum_required(VERSION 3.10)
project(Pluribus C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
find_package(Catch2 3 REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Boost REQUIRED)

link_directories("/usr/local/lib")
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(hand_isomorphism/)
add_subdirectory(omp/)
add_subdirectory(pluribus/)
add_subdirectory(test/)
add_subdirectory(benchmark/)

add_executable(Pluribus pluribus/main.cpp)
target_link_libraries(Pluribus PRIVATE OpenMP::OpenMP_CXX cnpy z PluribusLib HandIsoLib OMPEvalLib ${Boost_LIBRARIES})

add_executable(Temp temp/temp.cpp)
target_link_libraries(Temp PRIVATE OpenMP::OpenMP_CXX cnpy z PluribusLib HandIsoLib OMPEvalLib ${Boost_LIBRARIES})

add_executable(Test test/test.cpp)
target_link_libraries(Test PRIVATE OpenMP::OpenMP_CXX cnpy z PluribusLib HandIsoLib OMPEvalLib Catch2::Catch2WithMain ${Boost_LIBRARIES})

add_executable(Benchmark benchmark/benchmark.cpp)
target_link_libraries(Benchmark PRIVATE OpenMP::OpenMP_CXX cnpy z PluribusLib HandIsoLib OMPEvalLib Catch2::Catch2WithMain ${Boost_LIBRARIES})