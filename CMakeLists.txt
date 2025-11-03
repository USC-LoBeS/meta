# Minimum required CMake version
CMAKE_MINIMUM_REQUIRED(VERSION 3.15)

# Required CMake version
if(COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW)  
  cmake_policy(SET CMP0077 NEW)
  cmake_policy(SET CMP0042 NEW)
  cmake_policy(SET CMP0167 NEW)
endif()

# Project name and version
project(CMREP VERSION 1.0.0)

# Use C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build shared libraries
set(BUILD_SHARED_LIBS ON)

# # Get Eigen FIRST
find_package(Eigen3 CONFIG REQUIRED)

## Get LibXml2
# find_package(LibXml2 REQUIRED)
# include_directories(${LibXml2_INCLUDE_DIRS})
# link_libraries(${LibXml2_LIBRARIES})

# Get ITK and VTK
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})
# set(VTK_MODULE_USE_EXTERNAL_LibXml2 ON)
find_package(VTK 9 REQUIRED)

# Where to get additional modules
SET(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMake/")

# Set the include directories
include_directories(${CMREP_SOURCE_DIR}/src ${CMREP_SOURCE_DIR}/src/dijkstra)

# List of libraries needed to compile different components of CMREP
SET(CMREP_FIT_LIBS ${ITK_LIBRARIES} ${VTK_LIBRARIES})

## Sources for the PDE executable
set(COMMON_SRCS
  src/BranchingSubdivisionSurface.cxx
  src/BruteForceSubdivisionMedialModel.cxx
  src/GeometryDescriptor.cxx
  src/MedialAtom.cxx
  src/MedialAtomGrid.cxx
  src/MeshMedialPDESolver.cxx
  src/MeshTraversal.cxx
  src/Registry.cxx
  src/SparseMatrix.cxx
  src/SparseSolver.cxx            
  src/SubdivisionMedialModel.cxx
  src/SubdivisionSurface.cxx
  )

## Create the CM-REP library
add_library(cmrep ${COMMON_SRCS})
target_link_libraries(cmrep ${VTK_LIBRARIES})
vtk_module_autoinit(TARGETS cmrep MODULES ${VTK_LIBRARIES})

## Source code for the Dijkstra library
set(DIJKSTRA_SRCS src/dijkstra/ShortestPath.cxx src/dijkstra/VTKMeshShortestDistance.cxx)

## Create library of shortest path routines
add_library(cmrep_dijkstra ${DIJKSTRA_SRCS})
target_link_libraries(cmrep_dijkstra ${VTK_LIBRARIES})
vtk_module_autoinit(TARGETS cmrep_dijkstra MODULES ${VTK_LIBRARIES})

## Create vskel executable
add_executable(cmrep_vskel src/VoronoiSkeletonTool.cxx src/ReadWriteVTK.cxx)
target_link_libraries(cmrep_vskel cmrep cmrep_dijkstra ${CMREP_FIT_LIBS})

## Create levelset executable
add_executable(vtklevelset src/RealImageToMesh.cxx src/ReadWriteVTK.cxx)
target_link_libraries(vtklevelset cmrep ${CMREP_FIT_LIBS})


# Install rules
install(TARGETS cmrep cmrep_dijkstra cmrep_vskel vtklevelset
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)
