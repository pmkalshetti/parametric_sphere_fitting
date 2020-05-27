cmake_minimum_required(VERSION 3.12)

project(my_implementation)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)

# Eigen
set(EIGEN_INCLUDE_DIRS "./eigen_pr")

# Open3D
list(APPEND CMAKE_INSTALL_PREFIX "~/libraries/")
find_package(Open3D HINTS ${CMAKE_INSTALL_PREFIX}/lib/CMake)
list(APPEND Open3D_LIBRARIES dl)

# OpenSubdiv
set(OSD_INCLUDE_DIRS "/usr/local/include")
set(OSD_LIBRARY_DIRS "/usr/local/lib")
set(OSD_LIBS "${OSD_LIBRARY_DIRS}/libosdCPU.a")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${Open3D_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${Open3D_EXE_LINKER_FLAGS}")

include_directories(${EIGEN_INCLUDE_DIRS} ${Open3D_INCLUDE_DIRS} ${OSD_INCLUDE_DIRS})
link_directories(${Open3D_LIBRARY_DIRS} ${OSD_LIBRARY_DIRS})

add_executable(
    fit_sphere
    fit_sphere.cpp
    mesh.cpp
    subdiv_evaluator.cpp
    fitting_functor.cpp
)
target_link_libraries(fit_sphere ${Open3D_LIBRARIES} ${OSD_LIBS})