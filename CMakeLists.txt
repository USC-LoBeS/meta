# Minimum required CMake version
cmake_minimum_required(VERSION 3.12)

# Project name and version
project(CMREP VERSION 1.0.0 LANGUAGES CXX)

# Use C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build shared libraries
set(BUILD_SHARED_LIBS ON)

# Get ITK
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

# Get VTK
# find_package(VTK 9 REQUIRED COMPONENTS
#   CommonCore
#   CommonDataModel
#   CommonMath
#   CommonTransforms
#   FiltersCore
#   FiltersGeneral
#   FiltersGeometry
#   FiltersModeling
#   FiltersSources
#   FiltersVerdict
#   IOGeometry
#   IOImage
#   IOLegacy
#   IOPLY)
find_package(VTK 9 REQUIRED)

# Set the include directories
include_directories(${CMREP_SOURCE_DIR}/src 
                    ${CMREP_SOURCE_DIR}/src/dijkstra
)

# Get additional modules
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMake/")

# List of libraries needed to compile different components of CMREP
SET(CMREP_FIT_LIBS ${ITK_LIBRARIES} ${VTK_LIBRARIES})

# Source files for common library
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

# Create the CM-REP library
add_library(cmrep ${COMMON_SRCS})
target_link_libraries(cmrep ${VTK_LIBRARIES})
vtk_module_autoinit(TARGETS cmrep MODULES ${VTK_LIBRARIES})

# Source files for Dijkstra library
set(DIJKSTRA_SRCS
  src/dijkstra/ShortestPath.cxx
  src/dijkstra/VTKMeshShortestDistance.cxx
)

# Create library of shortest path routines
add_library(cmrep_dijkstra ${DIJKSTRA_SRCS})
target_link_libraries(cmrep_dijkstra ${VTK_LIBRARIES})
vtk_module_autoinit(TARGETS cmrep_dijkstra MODULES ${VTK_LIBRARIES})

## Create vskel executable
add_executable(cmrep_vskel src/VoronoiSkeletonTool.cxx src/ReadWriteVTK.cxx)
target_link_libraries(cmrep_vskel cmrep cmrep_dijkstra ${CMREP_FIT_LIBS})

## Create levelset executable
add_executable(vtklevelset src/RealImageToMesh.cxx src/ReadWriteVTK.cxx)
target_link_libraries(vtklevelset cmrep ${CMREP_FIT_LIBS})

# Ensure that the RPATH is set correctly
if(APPLE)
  set_target_properties(cmrep PROPERTIES INSTALL_RPATH "@loader_path/../lib")
  set_target_properties(cmrep_dijkstra PROPERTIES INSTALL_RPATH "@loader_path/../lib")
  set_target_properties(cmrep_vskel PROPERTIES INSTALL_RPATH "@loader_path/../lib")
  set_target_properties(vtklevelset PROPERTIES INSTALL_RPATH "@loader_path/../lib")
else()
  set_target_properties(cmrep PROPERTIES INSTALL_RPATH "\$ORIGIN/../lib")
  set_target_properties(cmrep_dijkstra PROPERTIES INSTALL_RPATH "\$ORIGIN/../lib")
  set_target_properties(cmrep_vskel PROPERTIES INSTALL_RPATH "\$ORIGIN/../lib")
  set_target_properties(vtklevelset PROPERTIES INSTALL_RPATH "\$ORIGIN/../lib")
endif()

# Install rules
install(TARGETS cmrep cmrep_dijkstra cmrep_vskel vtklevelset
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)
