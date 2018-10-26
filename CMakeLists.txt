cmake_minimum_required(VERSION 3.0)

project(mkobs3d CXX)
# SET(CMAKE_BUILD_TYPE Debug) # Seems CMAKE does not recognize -DCMAKE_BUILD_TYPE
set (mkobs3d_VERSION_MAJOR 0)
set (mkobj3d_VERSION_MINOR 2)

FIND_PACKAGE(PkgConfig REQUIRED)

message(STATUS "Initial CMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}")
message(STATUS "Initial Python links to: ${PYTHON_LIBRARY}")

if (APPLE)
  link_directories(
    "/usr/local/lib"
    "/opt/local/lib"
  )
endif()

include_directories(
  "/usr/local/include"
  "/opt/local/include"
  "/usr/local/Cellar/glew/2.0.0/include/"
  "/Users/randallsmith/Documents/sdks/git/libigl/include"
  "/lusr/include")
link_directories("$ENV{HOME}/.local/lib")

FIND_PACKAGE(OpenMP)

SET(USE_GPU TRUE)

find_package(OpenGL QUIET)
if (OpenGL_FOUND)
	include_directories(${OpenGL_INCLUDE_DIRS})
	link_directories(${OpenGL_LIBRARY_DIRS})
else (OpenGL_FOUND)
	SET(USE_GPU FALSE)
endif (OpenGL_FOUND)

find_package(GLEW QUIET)
if (GLEW_FOUND)
	include_directories(${GLEW_INCLUDE_DIRS})
	link_libraries(${GLEW_LIBRARIES})
else (GLEW_FOUND)
	SET(USE_GPU FALSE)
endif (GLEW_FOUND)

# GLFW
pkg_search_module(GLFW3 QUIET glfw3)
INCLUDE_DIRECTORIES(${GLFW3_INCLUDE_DIRS})
if (GLFW3_FOUND)
	SET(GLFW3_LIBRARY ${GLFW3_LIBRARIES})
	MESSAGE(STATUS "GLFW: ${GLFW3_LIBRARIES}")
else (GLFW3_FOUND)
	SET(USE_GPU FALSE)
endif(GLFW3_FOUND)

pkg_search_module(ODE QUIET ode-double)
if (NOT ODE_FOUND)
	pkg_search_module(ODE REQUIRED ode)
endif (NOT ODE_FOUND)

message("ODE_INCLUDE_DIRS ${ODE_INCLUDE_DIRS}")
message("ODE_CFLAGS ${ODE_CFLAGS}")

message(STATUS "Using GPU ${USE_GPU}")

# Include Eigen3 and IGL
INCLUDE(${CMAKE_SOURCE_DIR}/cmake/FindSUITESPARSE.cmake)
INCLUDE_DIRECTORIES(${SUITESPARSE_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(lib/)

FIND_PACKAGE(Eigen3 QUIET)
INCLUDE_DIRECTORIES(SYSTEM ${EIGEN3_INCLUDE_DIR})
INCLUDE_DIRECTORIES(${IGL_INCLUDE_DIR})


if (APPLE)
	  find_library(COCOA_LIBRARY Cocoa REQUIRED)
endif(APPLE)

# OMPL library shall be installed to default place
# Do NOT use pkg_search_module, it pulls too many dependencies and requires libfcl installed, while we use bundled libfcl.
# pkg_search_module(ompl REQUIRED ompl)
# MESSAGE(STATUS "OMPL INC: ${ompl_INCLUDE_DIRS}")
# INCLUDE_DIRECTORIES(${ompl_INCLUDE_DIRS})

# C++11 and Debug
# Note: we focus on gcc/clang right now, so we uses compiler specific solutions.
message(STATUS "BEFORE C++14 CMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "--std=c++14 -fdiagnostics-color=always ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${OpenMP_CXX_FLAGS} ${CMAKE_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "-g")
message(STATUS "AFTER C++14 CMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}")

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

FUNCTION(SANCHECK place)
	AUX_SOURCE_DIRECTORY(sancheck/${place}/ SAN_CHECK_SRC)
	ADD_EXECUTABLE(sancheck_${place} ${SAN_CHECK_SRC})
	FINALIZE(sancheck_${place})
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

FIND_PACKAGE(Boost 1.54.0 REQUIRED COMPONENTS timer system thread)
MESSAGE("Boost include dir: ${Boost_INCLUDE_DIR}")
MESSAGE("Boost libraries: ${Boost_LIBRARIES}")
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIR})
message(STATUS "AFTER Boost CMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}")
FIND_PACKAGE(Threads REQUIRED)
SET(CMAKE_THREAD_LIBS_INIT Threads::Threads)
MESSAGE("Threads: ${CMAKE_THREAD_LIBS_INIT}")
message(STATUS "AFTER Thread CMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}")

# CGAL Mess
# Can be disabled
option(MKOBS_USE_CGAL ON)
if (MKOBS_USE_CGAL)
	FIND_PACKAGE(CGAL QUIET COMPONENTS Common Core)
	message(STATUS "AFTER CGAL CMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}")
	SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CGAL_MODULES_DIR}")
endif (MKOBS_USE_CGAL)

FIND_PACKAGE(GMP QUIET)
message(STATUS "AFTER GMP CMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}")
FIND_PACKAGE(MPFR QUIET)
message(STATUS "AFTER MPFR CMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}")

# set(CGAL_LIB "${CGAL_LIB};CGAL")
message(STATUS "CGAL USE FILE ${CGAL_USE_FILE}")
message(STATUS "CGAL LIBRARIES ${CGAL_LIBRARIES}")
message(STATUS "CGAL LIBRARY ${CGAL_LIBRARY}")
set(CGAL_LIB "${CGAL_LIBRARY};${CGAL_Core_LIBRARY}")
message(STATUS "CGAL LIB ${CGAL_LIB}")

message(STATUS "CGAL_FOUND: ${CGAL_FOUND}")
IF (NOT CGAL_FOUND)
	message(WARNING "CGAL NOT FOUND, Some functions and libraires will not be built")
	add_definitions(-DPYOSR_NO_CGAL)
ENDIF (NOT CGAL_FOUND)


# FCL (Flexible Collision Library) support
SET(FCL_BUILD_TESTS OFF CACHE BOOL "Disable testcases for fcl" FORCE)
# SET(CMAKE_BUILD_TYPE Debug)
ADD_SUBDIRECTORY(third-party/fcl/)
# Prioritize bundled fcl.
INCLUDE_DIRECTORIES(BEFORE SYSTEM ${CMAKE_BINARY_DIR}/third-party/fcl/include/)
INCLUDE_DIRECTORIES(BEFORE SYSTEM ${CMAKE_SOURCE_DIR}/third-party/fcl/include/)

# SET(CMAKE_BUILD_TYPE Debug)

# Main program
ADD_EXECUTABLE(blend src/boolean/main.cc)
ADD_EXECUTABLE(subtract src/boolean/substract.cc)
FINALIZE(blend)
FINALIZE(subtract)

IF (CGAL_FOUND)
	EASYLIB(meshbool)
	TARGET_LINK_LIBRARIES(meshbool ${CGAL_LIB} ${CMAKE_THREAD_LIBS_INIT} ${GMP_LIBRARIES} ${MPFR_LIBRARIES} ${Boost_LIBRARIES})
	SET(BOOLEAN_LIB meshbool)
	TARGET_LINK_LIBRARIES(blend ${BOOLEAN_LIB})
	TARGET_LINK_LIBRARIES(subtract ${BOOLEAN_LIB})
ENDIF (CGAL_FOUND)

EASYLIB(tetio)
EASYLIB(vertcolorply)
EASYLIB(heatio)
EASYLIB(vecio)
EASYLIB(geopick)
EASYLIB(mazeinfo)
EASYLIB(omplaux)
EASYLIB(goct)
EASYLIB(tritri)

TARGET_LINK_LIBRARIES(mazeinfo vertcolorply)

if (USE_GPU)
EASYLIB(renderer)
TARGET_LINK_LIBRARIES(renderer ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES} ${GLFW3_LIBRARY})
endif(USE_GPU)

EASYADD(meshgen)
TARGET_LINK_LIBRARIES(meshgen mazeinfo)

EASYADD(omplgen)
TARGET_LINK_LIBRARIES(omplgen mazeinfo)

EASYADD(mkgen)
FILTERADD(flipfaces)
FILTERADD(addbb)
IF (CGAL_FOUND)
	FILTERADD(addcover)
	FILTERADD(hollow)
	TARGET_LINK_LIBRARIES(addcover meshbool)
	TARGET_LINK_LIBRARIES(hollow meshbool)
ENDIF (CGAL_FOUND)
EASYADD(dlap)
TARGET_LINK_LIBRARIES(dlap ${CMAKE_THREAD_LIBS_INIT} ${Boost_LIBRARIES} tetio)

EASYADD(Bcond)
TARGET_LINK_LIBRARIES(Bcond tetio)

EASYADD(NBcond)
TARGET_LINK_LIBRARIES(NBcond tetio vecio)

SANCHECK(laplacian)
TARGET_LINK_LIBRARIES(sancheck_laplacian tetio)

EASYADD(ring1picker)
TARGET_LINK_LIBRARIES(ring1picker tetio)

INCLUDE_DIRECTORIES(/opt/intel/mkl/include)
LINK_DIRECTORIES(/opt/intel/mkl/lib/intel64_lin/)
LINK_DIRECTORIES(/opt/intel/compilers_and_libraries/linux/lib/intel64/)
#LINK_DIRECTORIES(/usr/lib64/openmpi/lib/)
EASYADD(heat)
#TARGET_LINK_LIBRARIES(heat ${SUITESPARSE_LIBRARIES} vecio mkl_core mkl_intel_thread mkl_intel_lp64 iomp5 ptscotch scotch scotcherr pastix mpi z)
#TARGET_LINK_LIBRARIES(heat ${SUITESPARSE_LIBRARIES} vecio mkl_core mkl_intel_thread mkl_intel_lp64 iomp5)
message(STATUS "SUITESPARSE LIBS: ${SUITESPARSE_LIBRARIES}")
TARGET_LINK_LIBRARIES(heat ${SUITESPARSE_LIBRARIES} vecio)
#TARGET_INCLUDE_DIRECTORIES(heat BEFORE PRIVATE /usr/include/openmpi-x86_64/)
#TARGET_INCLUDE_DIRECTORIES(heat BEFORE PRIVATE /usr/local/include/pastix/)
#TARGET_INCLUDE_DIRECTORIES(heat BEFORE PRIVATE ${CMAKE_SOURCE_DIR}/third-party/eigen)

EASYADD(mass)
TARGET_LINK_LIBRARIES(mass tetio vecio)

if (USE_GPU)
EASYADD(visheat)
TARGET_LINK_LIBRARIES(visheat vertcolorply ${CMAKE_THREAD_LIBS_INIT} ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES} ${GLFW3_LIBRARY} tetio heatio)
endif (USE_GPU)

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

EASYLIB(erocol)
TARGET_LINK_LIBRARIES(erocol fat)
TARGET_LINK_LIBRARIES(omplaux erocol)
TARGET_LINK_LIBRARIES(omplaux tetio)
#
# All visualizer required GPU
#
if (USE_GPU)

SET(viscommon renderer ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES} ${GLFW3_LIBRARY})

EASYADD(freecube)
TARGET_LINK_LIBRARIES(freecube fcl omplaux ${CMAKE_THREAD_LIBS_INIT} ${viscommon})

SET(octbuildercommon goct vecio fcl omplaux ${CMAKE_THREAD_LIBS_INIT})

EASYADD(octbuilder)
TARGET_LINK_LIBRARIES(octbuilder ${octbuildercommon} ${viscommon})

EASYADD(naivebuilder)
TARGET_LINK_LIBRARIES(naivebuilder ${octbuildercommon} ${viscommon})

EASYADD(tbuilder)
TARGET_LINK_LIBRARIES(tbuilder ${octbuildercommon} ${viscommon} erocol)
endif (USE_GPU)

FIND_PACKAGE(vhacd QUIET)
IF ($(vhacd_FOUND))
EASYADD(convexpp)
TARGET_LINK_LIBRARIES(convexpp vhacd ${CMAKE_THREAD_LIBS_INIT} ${GMP_LIBRARIES} ${MPFR_LIBRARIES} ${CGAL_LIB} ${Boost_LIBRARIES})
ELSE ($(vhacd_FOUND))
	MESSAGE(STATUS "convexpp is NOT BUILT")
ENDIF ($(vhacd_FOUND))

EASYLIB(fat)
TARGET_LINK_LIBRARIES(fat openvdb tbb Half)

EASYADD(levelset)
TARGET_LINK_LIBRARIES(levelset fat)

SANCHECK(fatten)
PKG_SEARCH_MODULE(CCD REQUIRED ccd)
INCLUDE_DIRECTORIES(${CCD_INCLUDE_DIRS})

SANCHECK(ccd)
TARGET_LINK_LIBRARIES(sancheck_ccd ${CCD_LIBRARIES})

EASYLIB(ccdxx)
TARGET_LINK_LIBRARIES(omplaux ccdxx)

# TODO: remove deps on fcl
EASYADD(trmap)
TARGET_LINK_LIBRARIES(trmap omplaux fcl)

# Off-Screen Renderer for RL (reinforcement learning)
# However if OpenGL is disabled, then the renderer part is not compiled.
ADD_DEFINITIONS(-DGLM_ENABLE_EXPERIMENTAL -DGLM_FORCE_SIZE_FUNC=1 -DGLM_FORCE_RADIANS=1)
EASYLIB(osr)
TARGET_LINK_LIBRARIES(osr assimp fcl ${Boost_LIBRARIES} ompl tritri)
## Link osr to ode
TARGET_LINK_LIBRARIES(osr ${ODE_LIBRARIES})
TARGET_INCLUDE_DIRECTORIES(osr BEFORE PRIVATE ${ODE_INCLUDE_DIRS})
TARGET_COMPILE_DEFINITIONS(osr PRIVATE ${ODE_CFLAGS}) # Required, float or double
## Optional rendering function
if (USE_GPU)
TARGET_COMPILE_DEFINITIONS(osr PRIVATE GPU_ENABLED=1)
TARGET_LINK_LIBRARIES(osr gbm EGL ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES} ${Boost_LIBRARIES})

FIND_PACKAGE(PNG REQUIRED)
TARGET_LINK_LIBRARIES(osr ${PNG_LIBRARIES})
if (TARGET meshbool) # Only link if meshbool exists
TARGET_LINK_LIBRARIES(osr meshbool)
endif (TARGET meshbool)

### sancheck_osr checks its rendering functions
SANCHECK(osr)
TARGET_LINK_LIBRARIES(sancheck_osr osr)
TARGET_COMPILE_DEFINITIONS(sancheck_osr PRIVATE GPU_ENABLED=1)
else (USE_GPU)
TARGET_COMPILE_DEFINITIONS(osr PRIVATE GPU_ENABLED=0)
endif (USE_GPU)

# Blender PRM and RRT for RL's GT (Ground Truth)
EASYADD(mtblender)
## Note: do not use ompl_LIBRARIES, which pulls into too many denpendencies.
TARGET_LINK_LIBRARIES(mtblender ompl fcl)

EASYADD(maze2to3)

SANCHECK(ode)
TARGET_LINK_LIBRARIES(sancheck_ode ${ODE_LIBRARIES})
TARGET_INCLUDE_DIRECTORIES(sancheck_ode BEFORE PRIVATE ${ODE_INCLUDE_DIRS})
TARGET_COMPILE_DEFINITIONS(sancheck_ode PRIVATE ${ODE_CFLAGS}) # Required, float or double

# PYOSR
# FIND_PACKAGE(pybind11 REQUIRED)
find_package(PythonLibs 2.7 REQUIRED)

message(STATUS "Python version string: ${PYTHONLIBS_VERSION_STRING}")
AUX_SOURCE_DIRECTORY(lib/pyosr/ EASY_LOCAL_SRC)
EASYLIB(pyosr)
set_target_properties(pyosr PROPERTIES PREFIX "")
#pybind11_add_module(pyosr MODULE ${EASY_LOCAL_SRC})
message(STATUS "Python include directory: ${PYTHON_INCLUDE_DIR}")
TARGET_INCLUDE_DIRECTORIES(pyosr BEFORE PRIVATE ${PYTHON_INCLUDE_DIR})
message(STATUS "Python library: ${PYTHON_LIBRARY}")
TARGET_LINK_LIBRARIES(pyosr osr)

if (USE_GPU)
TARGET_COMPILE_DEFINITIONS(pyosr PRIVATE GPU_ENABLED=1)
else (USE_GPU)
TARGET_COMPILE_DEFINITIONS(osr PRIVATE GPU_ENABLED=0)
endif (USE_GPU)

SANCHECK(fclcontacts)
TARGET_LINK_LIBRARIES(sancheck_fclcontacts fcl)

# vim: tw=0
