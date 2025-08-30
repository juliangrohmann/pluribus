cmake_minimum_required(VERSION 3.10)
project(Pluribus C CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})

set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g -fno-omit-frame-pointer -fno-optimize-sibling-calls -DNDEBUG")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(VERBOSE)
  add_compile_definitions(VERBOSE)
endif()
if(UNIT_TEST)
  add_compile_definitions(UNIT_TEST)
endif()

find_package(Catch2 3 REQUIRED)

link_directories("/usr/local/lib")
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/libwandb/include)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/libwandb/lib)
link_libraries(wandb_cpp wandb_core)
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath,${CMAKE_SOURCE_DIR}/libwandb/lib")

add_compile_definitions(PROJECT_ROOT_DIR="${CMAKE_SOURCE_DIR}")

add_subdirectory(hand_isomorphism/)
add_subdirectory(omp/)
add_subdirectory(pluribus/)
add_subdirectory(test/)
add_subdirectory(benchmark/)

add_executable(Pluribus pluribus/main.cpp)
target_link_libraries(Pluribus PRIVATE PluribusLib)

add_executable(Temp temp/temp.cpp)
target_link_libraries(Temp PRIVATE PluribusLib)

if(UNIT_TEST) 
  add_executable(Test test/test.cpp)
  target_link_libraries(Test PRIVATE TestLib Catch2::Catch2WithMain)
  add_library(
          TestLib
          test/lib.cpp
  )
  target_link_libraries(TestLib PUBLIC PluribusLib)

  if(TESTSET)
    add_executable(TestSet test/testset.cpp)
    target_link_libraries(TestSet PRIVATE TestLib)
  endif()
endif()

add_executable(Benchmark benchmark/benchmark.cpp)
target_link_libraries(Benchmark PRIVATE PluribusLib Catch2::Catch2WithMain)
