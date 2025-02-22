cmake_minimum_required(VERSION 3.10)
project(Pluribus C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(VERBOSE)
  add_compile_definitions(VERBOSE)
endif()
if(UNIT_TEST)
  add_compile_definitions(UNIT_TEST)
endif()

find_package(Catch2 3 REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Boost REQUIRED)
find_package(TBB REQUIRED)

link_directories("/usr/local/lib")
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(hand_isomorphism/)
add_subdirectory(omp/)
add_subdirectory(pluribus/)
add_subdirectory(test/)
add_subdirectory(benchmark/)

add_executable(Pluribus pluribus/main.cpp)
target_link_libraries(Pluribus PRIVATE TBB::tbb OpenMP::OpenMP_CXX cnpy z PluribusLib HandIsoLib OMPEvalLib ${Boost_LIBRARIES})

add_executable(Temp temp/temp.cpp)
target_link_libraries(Temp PRIVATE TBB::tbb OpenMP::OpenMP_CXX cnpy z PluribusLib HandIsoLib OMPEvalLib ${Boost_LIBRARIES})

if(UNIT_TEST) 
  add_executable(Test test/test.cpp)
  target_link_libraries(Test PRIVATE TBB::tbb OpenMP::OpenMP_CXX cnpy z PluribusLib HandIsoLib OMPEvalLib Catch2::Catch2WithMain ${Boost_LIBRARIES})
endif()

add_executable(Benchmark benchmark/benchmark.cpp)
target_link_libraries(Benchmark PRIVATE TBB::tbb OpenMP::OpenMP_CXX cnpy z PluribusLib HandIsoLib OMPEvalLib Catch2::Catch2WithMain ${Boost_LIBRARIES})