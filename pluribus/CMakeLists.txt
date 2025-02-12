add_library(
  PluribusLib
  poker.cpp
  cluster.cpp
  util.cpp
  debug.cpp
)
target_link_libraries(PluribusLib PRIVATE OpenMP::OpenMP_CXX cnpy z HandIsoLib OMPEvalLib ${Boost_LIBRARIES})