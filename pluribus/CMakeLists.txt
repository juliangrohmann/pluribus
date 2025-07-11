find_package(OpenMP REQUIRED)
find_package(TBB REQUIRED)
find_package(SDL2 REQUIRED)
find_package(SDL2_image REQUIRED)
find_package(GSL REQUIRED)

find_package(PkgConfig REQUIRED)
pkg_check_modules(SDL2_TTF REQUIRED SDL2_ttf)

add_library(
  PluribusLib
  poker.cpp
  cluster.cpp
  agent.cpp
  simulate.cpp
  actions.cpp
  indexing.cpp
  mccfr.cpp
  pluribus.cpp
  blueprint.cpp
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