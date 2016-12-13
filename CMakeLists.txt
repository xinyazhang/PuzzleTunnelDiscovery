cmake_minimum_required(VERSION 3.0)

project(mkobs3d)
set (mkobs3d_VERSION_MAJOR 0)
set (mkobj3d_VERSION_MINOR 2)

FIND_PACKAGE(OpenMP)

# Include Eigen3 and IGL
FIND_PACKAGE(Eigen3 REQUIRED)
INCLUDE_DIRECTORIES(SYSTEM ${EIGEN3_INCLUDE_DIR})
INCLUDE_DIRECTORIES(${IGL_INCLUDE_DIR})

INCLUDE(${CMAKE_SOURCE_DIR}/cmake/FindSUITESPARSE.cmake)
INCLUDE_DIRECTORIES(${SUITESPARSE_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(lib/)

# GLFW
FIND_PACKAGE(PkgConfig REQUIRED)
pkg_search_module(GLFW3 REQUIRED glfw3)
INCLUDE_DIRECTORIES(${GLFW3_INCLUDE_DIRS})
SET(GLFW3_LIBRARY ${GLFW3_LIBRARIES})
MESSAGE(STATUS "GLFW: ${GLFW3_LIBRARIES}")

# C++11 and Debug
# Note: we focus on gcc/clang right now, so we uses compiler specific solutions.
set(CMAKE_CXX_FLAGS "--std=c++14 -g -O2 -march=haswell -ggdb3 -Wno-ignored-attributes -Wno-deprecated-declarations -Wno-deprecated ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${OpenMP_CXX_FLAGS} ${CMAKE_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "-g")

# Define function to copy binaries to bin/
FUNCTION(FINALCOPY target files)
	add_custom_command(TARGET ${target} POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E copy ${files} ${CMAKE_SOURCE_DIR}/bin
	)
ENDFUNCTION()
FUNCTION(FINALIZE target)
	add_custom_command(TARGET ${target} POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${target}> ${CMAKE_SOURCE_DIR}/bin
	)
ENDFUNCTION()

FUNCTION(EASYADD place)
	AUX_SOURCE_DIRECTORY(src/${place}/ EASY_LOCAL_SRC)
	ADD_EXECUTABLE(${place} ${EASY_LOCAL_SRC})
	FINALIZE(${place})
ENDFUNCTION()

FUNCTION(EASYLIB place)
	AUX_SOURCE_DIRECTORY(lib/${place}/ EASY_LOCAL_SRC)
	ADD_LIBRARY(${place} SHARED ${EASY_LOCAL_SRC})
	INCLUDE_DIRECTORIES(lib/${place})
	FINALIZE(${place})
ENDFUNCTION()

FUNCTION(FILTERADD filter)
	SET(FT_LOCAL_SRC src/filters/${filter}.cc)
	ADD_EXECUTABLE(${filter} ${FT_LOCAL_SRC})
	FINALIZE(${filter})
ENDFUNCTION()

#THIRD PARTIES
# OBJ2PLY
#include(ExternalProject)
#SET(OBJ2PLY_BIN "${CMAKE_BINARY_DIR}/bin")
#SET(OBJ2PLY_ARGS "-DIGL_INCLUDE_DIR=${IGL_INCLUDE_DIR}")
#LIST(APPEND OBJ2PLY_ARGS "-DEIGEN3_INCLUDE_DIR=${EIGEN3_INCLUDE_DIR}")
#ExternalProject_Add(OBJ2PLY
#	GIT_REPOSITORY "https://github.com/xinyazhang/obj2ply.git"
#	GIT_SUBMODULES "${CMAKE_SOURCE_DIR}/third-party/obj2ply/"
#	CMAKE_ARGS ${OBJ2PLY_ARGS}
#	INSTALL_COMMAND ""
#	BINARY_DIR "${OBJ2PLY_BIN}")
#FINALCOPY(OBJ2PLY ${OBJ2PLY_BIN}/obj2ply)

FIND_PACKAGE(Boost 1.54.0 COMPONENTS timer)
MESSAGE("Boost include dir: ${Boost_INCLUDE_DIR}")
MESSAGE("Boost libraries: ${Boost_LIBRARIES}")
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIR})
FIND_PACKAGE(Threads REQUIRED)
FIND_PACKAGE(CGAL COMPONENTS Core REQUIRED)
INCLUDE(${CGAL_USE_FILE})
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CGAL_MODULES_DIR}")
FIND_PACKAGE(GMP REQUIRED)
FIND_PACKAGE(MPFR REQUIRED)

# FCL (Flexible Collision Library) support
ADD_SUBDIRECTORY(third-party/fcl/)
INCLUDE_DIRECTORIES(SYSTEM ${CMAKE_BINARY_DIR}/third-party/fcl/include/)
INCLUDE_DIRECTORIES(SYSTEM ${CMAKE_SOURCE_DIR}/third-party/fcl/include/)

# Main program
ADD_EXECUTABLE(blend src/boolean/main.cc)
ADD_EXECUTABLE(subtract src/boolean/substract.cc)
FINALIZE(blend)
FINALIZE(subtract)

EASYLIB(meshbool)
TARGET_LINK_LIBRARIES(meshbool ${CGAL_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${GMP_LIBRARIES} ${MPFR_LIBRARIES})
SET(BOOLEAN_LIB meshbool)
TARGET_LINK_LIBRARIES(blend ${BOOLEAN_LIB})
TARGET_LINK_LIBRARIES(subtract ${BOOLEAN_LIB})

EASYLIB(tetio)
EASYLIB(vertcolorply)
EASYLIB(heatio)
EASYLIB(vecio)
EASYLIB(geopick)
EASYLIB(mazeinfo)
EASYLIB(renderer)
EASYLIB(omplaux)

TARGET_LINK_LIBRARIES(mazeinfo vertcolorply)

EASYADD(meshgen)
TARGET_LINK_LIBRARIES(meshgen mazeinfo)

EASYADD(omplgen)
TARGET_LINK_LIBRARIES(omplgen mazeinfo)

EASYADD(mkgen)
FILTERADD(flipfaces)
FILTERADD(addbb)
FILTERADD(addcover)
FILTERADD(hollow)
TARGET_LINK_LIBRARIES(addcover meshbool)
TARGET_LINK_LIBRARIES(hollow meshbool)
EASYADD(dlap)
TARGET_LINK_LIBRARIES(dlap ${CMAKE_THREAD_LIBS_INIT} ${Boost_LIBRARIES} tetio)

EASYADD(Bcond)
TARGET_LINK_LIBRARIES(Bcond tetio)

EASYADD(NBcond)
TARGET_LINK_LIBRARIES(NBcond tetio vecio)

EASYADD(sancheck)
TARGET_LINK_LIBRARIES(sancheck tetio)

EASYADD(ring1picker)
TARGET_LINK_LIBRARIES(ring1picker tetio)

INCLUDE_DIRECTORIES(/opt/intel/mkl/include)
LINK_DIRECTORIES(/opt/intel/mkl/lib/intel64_lin/)
LINK_DIRECTORIES(/opt/intel/compilers_and_libraries/linux/lib/intel64/)
#LINK_DIRECTORIES(/usr/lib64/openmpi/lib/)
EASYADD(heat)
#TARGET_LINK_LIBRARIES(heat ${SUITESPARSE_LIBRARIES} vecio mkl_core mkl_intel_thread mkl_intel_lp64 iomp5 ptscotch scotch scotcherr pastix mpi z)
TARGET_LINK_LIBRARIES(heat ${SUITESPARSE_LIBRARIES} vecio mkl_core mkl_intel_thread mkl_intel_lp64 iomp5)
TARGET_LINK_LIBRARIES(heat ${SUITESPARSE_LIBRARIES} vecio)
#TARGET_INCLUDE_DIRECTORIES(heat BEFORE PRIVATE /usr/include/openmpi-x86_64/)
#TARGET_INCLUDE_DIRECTORIES(heat BEFORE PRIVATE /usr/local/include/pastix/)
#TARGET_INCLUDE_DIRECTORIES(heat BEFORE PRIVATE ${CMAKE_SOURCE_DIR}/third-party/eigen)

EASYADD(mass)
TARGET_LINK_LIBRARIES(mass tetio vecio)

find_package(GLEW REQUIRED)
find_package(OpenGL REQUIRED)
EASYADD(visheat)
TARGET_LINK_LIBRARIES(visheat vertcolorply ${CMAKE_THREAD_LIBS_INIT} ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES} ${GLFW3_LIBRARY} tetio heatio)

EASYADD(follow)
TARGET_LINK_LIBRARIES(follow tetio heatio)

EASYADD(invgen)

EASYADD(periodicalize)
TARGET_LINK_LIBRARIES(periodicalize tetio geopick)

EASYADD(tet2obj)
TARGET_LINK_LIBRARIES(tet2obj tetio geopick)
#FILE(COPY ${BINARIES} DESTINATION ${CMAKE_SOURCE_DIR}/bin)

EASYADD(mergep)
EASYADD(evadiff)
TARGET_LINK_LIBRARIES(evadiff tetio heatio)

#EASYADD(minkview2)
#TARGET_LINK_LIBRARIES(minkview2 ${CGAL_LIBRARIES} ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES} ${GLFW3_LIBRARY})

EASYADD(freecube)
TARGET_LINK_LIBRARIES(freecube fcl renderer omplaux ${CMAKE_THREAD_LIBS_INIT} ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES} ${GLFW3_LIBRARY})

EASYADD(octbuilder)
TARGET_LINK_LIBRARIES(octbuilder fcl omplaux ${CMAKE_THREAD_LIBS_INIT})
# vim: tw=0
