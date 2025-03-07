add_library(
  PluribusLib
  poker.cpp
  cluster.cpp
  agent.cpp
  simulate.cpp
  actions.cpp
  history_index.cpp
  infoset.cpp
  mccfr.cpp
  blueprint.cpp
  range.cpp
  range_viewer.cpp
  util.cpp
  debug.cpp
)
target_link_libraries(PluribusLib PRIVATE ${SDL2_LIBRARIES} ${SDL2_TTF_LIBRARIES} TBB::tbb OpenMP::OpenMP_CXX cnpy z HandIsoLib OMPEvalLib ${Boost_LIBRARIES})