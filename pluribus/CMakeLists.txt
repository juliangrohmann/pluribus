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
  storage.cpp
  util.cpp
  debug.cpp
)
target_link_libraries(PluribusLib PRIVATE TBB::tbb OpenMP::OpenMP_CXX cnpy z HandIsoLib OMPEvalLib ${Boost_LIBRARIES})