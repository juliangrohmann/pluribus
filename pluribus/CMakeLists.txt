find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(TBB REQUIRED)
find_package(SDL2 REQUIRED)
find_package(SDL2_image REQUIRED)
find_package(GSL REQUIRED)

find_package(PkgConfig REQUIRED)
pkg_check_modules(SDL2_TTF REQUIRED SDL2_ttf)

if(NOT DEFINED OpenMP_CXX_SPEC_DATE)
    message(FATAL_ERROR "CMake’s OpenMP module did not provide SPEC_DATE; upgrade CMake or use Option B.")
endif()
if(OpenMP_CXX_SPEC_DATE LESS 201307)
    message(FATAL_ERROR
            "OpenMP 4.0+ required. Detected spec date: ${OpenMP_CXX_SPEC_DATE} "
            "(version ${OpenMP_CXX_VERSION}).")
endif()

add_library(
  PluribusLib
  poker.cpp
  cluster.cpp
  earth_movers_dist.cpp
  agent.cpp
  simulate.cpp
  actions.cpp
  decision.cpp
  translate.cpp
  indexing.cpp
  mccfr.cpp
  pluribus.cpp
  blueprint.cpp
  server.cpp
  traverse.cpp
  range.cpp
  range_viewer.cpp
  sampling.cpp
  ev.cpp
  calc.cpp
  config.cpp
  profiles.cpp
  dist.cpp
  util.cpp
  debug.cpp
  logging.cpp
)
target_link_libraries(PluribusLib PUBLIC 
  ${SDL2_IMAGE_LIBRARIES} 
  ${SDL2_LIBRARIES} 
  ${SDL2_TTF_LIBRARIES} 
  ${GSL_LIBRARIES}
  TBB::tbb 
  OpenMP::OpenMP_CXX 
  cnpy 
  HandIsoLib 
  OMPEvalLib
)