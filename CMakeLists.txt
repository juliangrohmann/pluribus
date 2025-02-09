cmake_minimum_required(VERSION 3.10)
project(Pluribus)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_BUILD_TYPE Debug)
find_package(Catch2 3 REQUIRED)
find_package(Boost REQUIRED)

include_directories(${Boost_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(pluribus/)
add_subdirectory(omp/)
add_subdirectory(tests/)

add_executable(Pluribus pluribus/main.cpp)
target_link_libraries(Pluribus PRIVATE PluribusLib OMPEvalLib ${Boost_LIBRARIES})

add_executable(Temp temp/temp.cpp)
target_link_libraries(Temp PRIVATE PluribusLib OMPEvalLib ${Boost_LIBRARIES})

add_executable(Tests tests/tests.cpp)
target_link_libraries(Tests PRIVATE PluribusLib OMPEvalLib Catch2::Catch2WithMain ${Boost_LIBRARIES})