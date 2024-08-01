# Requisite CMake commands
cmake_minimum_required(VERSION 3.0)
if(COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

# Project name
PROJECT(CMREP)
# SET(CMAKE_CXX_STANDARD 11)

# Versioning
SET(CMREP_VERSION_MAJOR 1)
SET(CMREP_VERSION_MINOR 0)
SET(CMREP_VERSION_PATCH 0)
SET(CMREP_VERSION_FULL "${CMREP_VERSION_MAJOR}.${CMREP_VERSION_MINOR}.${CMREP_VERSION_PATCH}")

# Get ITK
FIND_PACKAGE(ITK REQUIRED)
INCLUDE(${ITK_USE_FILE})

# Get VTK
FIND_PACKAGE(VTK 9 REQUIRED)


# Print ITK/VTK path
message(STATUS "ITK_DIR: ${ITK_DIR}")
message(STATUS "VTK_DIR: ${VTK_DIR}")

# Where to get additional modules
SET(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMake/")

# Set the include directories
INCLUDE_DIRECTORIES(${CMREP_SOURCE_DIR}/src ${CMREP_SOURCE_DIR}/src/dijkstra)


# List of libraries needed to compile different components of CMREP
SET(CMREP_FIT_LIBS "")

INCLUDE(ExternalProject)

# Determine which solver wrapper to build for PDE
SET(CMREP_FIT_LIBS ${ITK_LIBRARIES} ${VTK_LIBRARIES})


# Sources for the PDE executable
SET(COMMON_SRCS
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
ADD_LIBRARY(cmrep ${COMMON_SRCS})
TARGET_LINK_LIBRARIES(cmrep ${VTK_LIBRARIES})
vtk_module_autoinit(TARGETS cmrep MODULES ${VTK_LIBRARIES})

## Source code for the Dijkstra library
SET(DIJKSTRA_SRCS src/dijkstra/ShortestPath.cxx src/dijkstra/VTKMeshShortestDistance.cxx)

## Create library of shortest path routines
ADD_LIBRARY(cmrep_dijkstra ${DIJKSTRA_SRCS})
TARGET_LINK_LIBRARIES(cmrep_dijkstra ${VTK_LIBRARIES})
vtk_module_autoinit(TARGETS cmrep_dijkstra MODULES ${VTK_LIBRARIES})

## Create vskel executable
ADD_EXECUTABLE(cmrep_vskel src/VoronoiSkeletonTool.cxx src/ReadWriteVTK.cxx)
TARGET_LINK_LIBRARIES(cmrep_vskel cmrep cmrep_dijkstra ${CMREP_FIT_LIBS})


## Create levelset executable
ADD_EXECUTABLE(vtklevelset src/RealImageToMesh.cxx src/ReadWriteVTK.cxx)
TARGET_LINK_LIBRARIES(vtklevelset cmrep ${CMREP_FIT_LIBS})

# Install rules
install(TARGETS cmrep cmrep_dijkstra cmrep_vskel vtklevelset
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)