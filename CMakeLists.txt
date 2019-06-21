cmake_minimum_required(VERSION 3.13)
# We need 3.13+ to support target_include_directories
# Alternative solutions (e.g. set_target_properties) cannot override the system's ompl library

project(mkobs3d CXX)
# SET(CMAKE_BUILD_TYPE Debug) # Seems CMAKE does not recognize -DCMAKE_BUILD_TYPE
set (mkobs3d_VERSION_MAJOR 0)
set (mkobs3d_VERSION_MINOR 5)

include(ProcessorCount)
ProcessorCount(NCPU)

# External project
include(ExternalProject)
set(EXTERNAL_PROJECTS_PREFIX ${CMAKE_BINARY_DIR}/external)
set(EXTERNAL_PROJECTS_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/installed)

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

SET(OpenGL_GL_PREFERENCE GLVND) # Prefer GLVND since all our platforms are using it
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

# Include lib/ so every app under src/ can use them
include_directories(${CMAKE_SOURCE_DIR}/lib/)

# Include Eigen3 and IGL
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

# C++14 and Debug
message(STATUS "BEFORE C++14 CMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
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
	#INCLUDE_DIRECTORIES(lib/${place})
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

FIND_PACKAGE(Boost 1.54.0 REQUIRED COMPONENTS system thread)
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
option(MKOBS_USE_CGAL "Enable functions that depend on CGAL" ON)
option(MKOBS_FORCE_CGAL "Make CGAL REQUIRED instead of optional" ON)
if (MKOBS_USE_CGAL)
	if (MKOBS_FORCE_CGAL)
		FIND_PACKAGE(CGAL REQUIRED COMPONENTS Common Core)
	else (MKOBS_FORCE_CGAL)
		FIND_PACKAGE(CGAL QUIET COMPONENTS Common Core)
	endif (MKOBS_FORCE_CGAL)
	message(STATUS "AFTER CGAL CMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}")
	#SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CGAL_MODULES_DIR}")
endif (MKOBS_USE_CGAL)

FIND_PACKAGE(GMP QUIET)
message(STATUS "AFTER GMP CMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}")
FIND_PACKAGE(MPFR QUIET)
message(STATUS "AFTER MPFR CMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}")

# set(CGAL_LIB "${CGAL_LIB};CGAL")
# message(STATUS "CGAL USE FILE ${CGAL_USE_FILE}")
# message(STATUS "CGAL LIBRARIES ${CGAL_LIBRARIES}")
# message(STATUS "CGAL LIBRARY ${CGAL_LIBRARY}")
# set(CGAL_LIB "${CGAL_LIBRARY};${CGAL_Core_LIBRARY}")
# message(STATUS "CGAL LIB ${CGAL_LIB}")

message(STATUS "CGAL_FOUND: ${CGAL_FOUND}")

ExternalProject_Add(ext_fcl
	PREFIX ${EXTERNAL_PROJECTS_PREFIX}
	SOURCE_DIR ${CMAKE_SOURCE_DIR}/third-party/fcl/
	CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release -DFCL_BUILD_TESTS=OFF -DCMAKE_INSTALL_PREFIX=${EXTERNAL_PROJECTS_INSTALL_PREFIX}
	BUILD_COMMAND make -j ${NCPU})
set(ENV{PKG_CONFIG_PATH} "$ENV{PKG_CONFIG_PATH}:${EXTERNAL_PROJECTS_INSTALL_PREFIX}/lib64/pkgconfig:${EXTERNAL_PROJECTS_INSTALL_PREFIX}/lib/pkgconfig")

function(use_fcl TGT)
	add_dependencies(${TGT} ext_fcl)
	target_include_directories("${TGT}" BEFORE PRIVATE ${EXTERNAL_PROJECTS_INSTALL_PREFIX}/include)
	target_link_directories("${TGT}" BEFORE PRIVATE ${EXTERNAL_PROJECTS_INSTALL_PREFIX}/lib ${EXTERNAL_PROJECTS_INSTALL_PREFIX}/lib64)
	target_link_libraries("${TGT}" PRIVATE fcl)
endfunction()

# SET(CMAKE_BUILD_TYPE Debug)

IF (CGAL_FOUND)
	message(STATUS "Find CGAL at\n\tHeaders:\t${CGAL_INCLUDE_DIRS}\n\tLibs: ${CGAL_LIB}")
	EASYLIB(meshbool)
	TARGET_INCLUDE_DIRECTORIES(meshbool BEFORE PRIVATE ${CGAL_INCLUDE_DIRS})
	TARGET_LINK_LIBRARIES(meshbool CGAL::CGAL CGAL::CGAL_Core ${CMAKE_THREAD_LIBS_INIT} ${GMP_LIBRARIES} ${MPFR_LIBRARIES} ${Boost_LIBRARIES})
ELSE (CGAL_FOUND)
	message(WARNING "CGAL NOT FOUND, Some functions and libraires will not be built")
ENDIF (CGAL_FOUND)

IF (TARGET meshbool)
	# Main program
	ADD_EXECUTABLE(blend src/boolean/main.cc)
	ADD_EXECUTABLE(subtract src/boolean/substract.cc)
	FINALIZE(blend)
	FINALIZE(subtract)
	SET(BOOLEAN_LIB meshbool)
	TARGET_LINK_LIBRARIES(blend ${BOOLEAN_LIB})
	TARGET_LINK_LIBRARIES(subtract ${BOOLEAN_LIB})

	FILTERADD(addcover)
	FILTERADD(hollow)
	TARGET_LINK_LIBRARIES(addcover meshbool)
	TARGET_LINK_LIBRARIES(hollow meshbool)
ENDIF (TARGET meshbool)

EASYLIB(tetio)
EASYLIB(advplyio)
EASYLIB(heatio)
EASYLIB(vecio)
EASYLIB(geopick)
EASYLIB(mazeinfo)
EASYLIB(goct)
EASYLIB(tritri)

EASYLIB(omplaux)
target_link_libraries(omplaux PRIVATE tetio)
use_fcl(omplaux)

TARGET_LINK_LIBRARIES(mazeinfo advplyio)

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
EASYADD(dlap)
TARGET_LINK_LIBRARIES(dlap ${CMAKE_THREAD_LIBS_INIT} ${Boost_LIBRARIES} tetio)

EASYADD(Bcond)
TARGET_LINK_LIBRARIES(Bcond tetio)

EASYADD(NBcond)
TARGET_LINK_LIBRARIES(NBcond tetio vecio)

EASYADD(ring1picker)
TARGET_LINK_LIBRARIES(ring1picker tetio)

INCLUDE_DIRECTORIES(/opt/intel/mkl/include)
LINK_DIRECTORIES(/opt/intel/mkl/lib/intel64_lin/)
LINK_DIRECTORIES(/opt/intel/compilers_and_libraries/linux/lib/intel64/)

find_package(SUITESPARSE QUIET)
if (${SUITESPARSE_FOUND})
#LINK_DIRECTORIES(/usr/lib64/openmpi/lib/)
	EASYADD(heat)
	target_include_directories(heat ${SUITESPARSE_INCLUDE_DIRS})
#TARGET_LINK_LIBRARIES(heat ${SUITESPARSE_LIBRARIES} vecio mkl_core mkl_intel_thread mkl_intel_lp64 iomp5 ptscotch scotch scotcherr pastix mpi z)
#TARGET_LINK_LIBRARIES(heat ${SUITESPARSE_LIBRARIES} vecio mkl_core mkl_intel_thread mkl_intel_lp64 iomp5)
	message(STATUS "SUITESPARSE LIBS: ${SUITESPARSE_LIBRARIES}")
	TARGET_LINK_LIBRARIES(heat ${SUITESPARSE_LIBRARIES} vecio)
#TARGET_INCLUDE_DIRECTORIES(heat BEFORE PRIVATE /usr/include/openmpi-x86_64/)
#TARGET_INCLUDE_DIRECTORIES(heat BEFORE PRIVATE /usr/local/include/pastix/)
#TARGET_INCLUDE_DIRECTORIES(heat BEFORE PRIVATE ${CMAKE_SOURCE_DIR}/third-party/eigen)
	SANCHECK(laplacian)
	target_link_libraries(sancheck_laplacian tetio)
else ()
	message("SuiteSparse is not found")
endif ()

EASYADD(mass)
TARGET_LINK_LIBRARIES(mass tetio vecio)

if (USE_GPU)
EASYADD(visheat)
TARGET_LINK_LIBRARIES(visheat advplyio ${CMAKE_THREAD_LIBS_INIT} ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES} ${GLFW3_LIBRARY} tetio heatio)
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

#
# All visualizer required GPU
#
if (USE_GPU)

SET(viscommon renderer ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES} ${GLFW3_LIBRARY})

EASYADD(freecube)
target_link_libraries(freecube PRIVATE omplaux ${CMAKE_THREAD_LIBS_INIT} ${viscommon} ccdxx)
use_fcl(freecube)

EASYLIB(ccdxx)
target_link_libraries(omplaux PRIVATE ccdxx)

SET(octbuildercommon goct vecio omplaux ${CMAKE_THREAD_LIBS_INIT})

EASYADD(octbuilder)
target_link_libraries(octbuilder PRIVATE ${octbuildercommon} ${viscommon} ccdxx)
use_fcl(octbuilder)

EASYADD(naivebuilder)
target_link_libraries(naivebuilder PRIVATE ${octbuildercommon} ${viscommon})
use_fcl(naivebuilder)

endif (USE_GPU)

FIND_PACKAGE(vhacd QUIET)
IF ($(vhacd_FOUND))
EASYADD(convexpp)
TARGET_LINK_LIBRARIES(convexpp vhacd ${CMAKE_THREAD_LIBS_INIT} ${GMP_LIBRARIES} ${MPFR_LIBRARIES} ${CGAL_LIB} ${Boost_LIBRARIES})
ELSE ($(vhacd_FOUND))
	MESSAGE(STATUS "convexpp is NOT BUILT")
ENDIF ($(vhacd_FOUND))

find_package(OpenVDB QUIET)
message("OpenVDB_FOUND: ${OpenVDB_FOUND}")
if (${OpenVDB_FOUND})
	EASYLIB(fat)
	target_include_directories(fat PUBLIC ${OpenVDB_INCLUDE_DIR})
	target_link_directories(fat PUBLIC ${OpenVDB_LIBRARY_DIR})
	target_link_libraries(fat ${OpenVDB_LIBRARIES} tbb)

	EASYADD(levelset)
	TARGET_LINK_LIBRARIES(levelset fat)

	EASYLIB(erocol)
	target_link_libraries(erocol fat)

	if (USE_GPU)
		EASYADD(tbuilder)
		target_link_libraries(tbuilder PRIVATE ${octbuildercommon} ${viscommon} erocol)
		use_fcl(tbuilder)
	endif (USE_GPU)

	target_link_libraries(omplaux erocol)
endif ()


SANCHECK(fatten)
PKG_SEARCH_MODULE(CCD REQUIRED ccd)
INCLUDE_DIRECTORIES(${CCD_INCLUDE_DIRS})

SANCHECK(ccd)
TARGET_LINK_LIBRARIES(sancheck_ccd ${CCD_LIBRARIES})
target_link_directories(sancheck_ccd PRIVATE ${CCD_LIBRARY_DIRS})
message("CCD_LIBRARIES ${CCD_LIBRARIES}")
message("CCD_LIBRARY_DIRS ${CCD_LIBRARY_DIRS}")

# TODO: remove deps on fcl
EASYADD(trmap)
target_link_directories(trmap PRIVATE ${CCD_LIBRARY_DIRS})
target_link_libraries(trmap PRIVATE omplaux)
use_fcl(trmap)

# Off-Screen Renderer for RL (reinforcement learning)
# However if OpenGL is disabled, then the renderer part is not compiled.
ADD_DEFINITIONS(-DGLM_ENABLE_EXPERIMENTAL -DGLM_FORCE_SIZE_FUNC=1 -DGLM_FORCE_RADIANS=1)
EASYLIB(osr)
target_link_libraries(osr PRIVATE assimp ${Boost_LIBRARIES} tritri)
use_fcl(osr)
target_link_libraries(osr PRIVATE ompl)

## Link osr to ode
target_link_directories(osr PUBLIC ${CCD_LIBRARY_DIRS})
target_link_libraries(osr PRIVATE ${ODE_LIBRARIES} ${CCD_LIBRARIES})
TARGET_INCLUDE_DIRECTORIES(osr BEFORE PRIVATE ${ODE_INCLUDE_DIRS})
TARGET_COMPILE_DEFINITIONS(osr PRIVATE ${ODE_CFLAGS}) # Required, float or double

## OPTIONAL FUNCTION 1: rendering
if (USE_GPU)
	TARGET_COMPILE_DEFINITIONS(osr PRIVATE GPU_ENABLED=1)
	target_link_libraries(osr PRIVATE gbm EGL ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES} ${Boost_LIBRARIES})
	FIND_PACKAGE(PNG REQUIRED)
	target_link_libraries(osr PRIVATE ${PNG_LIBRARIES})


### sancheck_osr checks its rendering functions
	SANCHECK(osr)
	TARGET_LINK_LIBRARIES(sancheck_osr osr)
	TARGET_COMPILE_DEFINITIONS(sancheck_osr PRIVATE GPU_ENABLED=1)

	EASYADD(vistexture)
	TARGET_COMPILE_DEFINITIONS(vistexture PRIVATE GPU_ENABLED=1)
	TARGET_LINK_LIBRARIES(vistexture osr ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES} ${GLFW3_LIBRARY})

	EASYADD(cfreeman)
	TARGET_COMPILE_DEFINITIONS(cfreeman PRIVATE GPU_ENABLED=1)
	TARGET_LINK_LIBRARIES(cfreeman osr ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES} ${GLFW3_LIBRARY})

	EASYADD(vispath)
	TARGET_COMPILE_DEFINITIONS(vispath PRIVATE GPU_ENABLED=1)
	TARGET_LINK_LIBRARIES(vispath osr ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES} ${GLFW3_LIBRARY})
else (USE_GPU)
	TARGET_COMPILE_DEFINITIONS(osr PRIVATE GPU_ENABLED=0)
endif (USE_GPU)

## OPTIONAL FUNCTION 2: meshbool
IF (CGAL_FOUND)
	TARGET_COMPILE_DEFINITIONS(osr PRIVATE PYOSR_HAS_CGAL=1)
else (CGAL_FOUND)
	TARGET_COMPILE_DEFINITIONS(osr PRIVATE PYOSR_HAS_CGAL=0)
endif (CGAL_FOUND)

if (TARGET meshbool) # Only link if meshbool exists
	target_link_libraries(osr PRIVATE meshbool)
endif (TARGET meshbool)

EASYLIB(rectpack)
use_fcl(rectpack)
EASYADD(objautouv)
TARGET_LINK_LIBRARIES(objautouv rectpack)

# Blender PRM and RRT for RL's GT (Ground Truth)
EASYADD(mtblender)
target_link_directories(mtblender PRIVATE ${CCD_LIBRARY_DIRS})
## Note: do not use ompl_LIBRARIES, which pulls into too many denpendencies.
target_link_libraries(mtblender PRIVATE ompl)
use_fcl(mtblender)

EASYADD(mapmerger)
target_link_directories(mapmerger PRIVATE ${CCD_LIBRARY_DIRS})
target_link_libraries(mapmerger PRIVATE ompl)
use_fcl(mapmerger)

EASYADD(maze2to3)

SANCHECK(ode)
TARGET_LINK_LIBRARIES(sancheck_ode ${ODE_LIBRARIES})
TARGET_INCLUDE_DIRECTORIES(sancheck_ode BEFORE PRIVATE ${ODE_INCLUDE_DIRS})
TARGET_COMPILE_DEFINITIONS(sancheck_ode PRIVATE ${ODE_CFLAGS}) # Required, float or double

# PYOSR
# FIND_PACKAGE(pybind11 REQUIRED)
find_package(PythonLibs 3 REQUIRED)
find_package(pybind11 REQUIRED)

message(STATUS "Python version string: ${PYTHONLIBS_VERSION_STRING}")
set(EASY_LOCAL_SRC)
aux_source_directory(lib/pyosr/ EASY_LOCAL_SRC)
pybind11_add_module(pyosr MODULE ${EASY_LOCAL_SRC})
#set_target_properties(pyosr PROPERTIES PREFIX "")
#message(STATUS "Python include directory: ${PYTHON_INCLUDE_DIR}")
TARGET_INCLUDE_DIRECTORIES(pyosr BEFORE PRIVATE ${PYTHON_INCLUDE_DIR})
#message(STATUS "Python library: ${PYTHON_LIBRARY}")
target_link_libraries(pyosr PRIVATE osr)
use_fcl(pyosr)
FINALIZE(pyosr)

if (USE_GPU)
	TARGET_COMPILE_DEFINITIONS(pyosr PRIVATE GPU_ENABLED=1)
else (USE_GPU)
	TARGET_COMPILE_DEFINITIONS(pyosr PRIVATE GPU_ENABLED=0)
endif (USE_GPU)

IF (CGAL_FOUND)
	TARGET_COMPILE_DEFINITIONS(pyosr PRIVATE PYOSR_HAS_CGAL=1)
else (CGAL_FOUND)
	TARGET_COMPILE_DEFINITIONS(pyosr PRIVATE PYOSR_HAS_CGAL=0)
endif (CGAL_FOUND)

SANCHECK(fclcontacts)
target_link_directories(sancheck_fclcontacts PRIVATE ${CCD_LIBRARY_DIRS})
target_link_libraries(sancheck_fclcontacts PRIVATE ${CCD_LIBRARIES})
use_fcl(sancheck_fclcontacts)

ExternalProject_Add(ext_ompl_app
	PREFIX ${EXTERNAL_PROJECTS_PREFIX}
	SOURCE_DIR ${CMAKE_SOURCE_DIR}/third-party/ompl.app/
	CMAKE_ARGS -E env PKG_CONFIG_PATH="$ENV{PKG_CONFIG_PATH}" ${CMAKE_COMMAND} -DOMPL_BUILD_DEMOS=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${EXTERNAL_PROJECTS_INSTALL_PREFIX}
	BUILD_COMMAND make -j ${NCPU} ompl ompl_app_base
	BUILD_ALWAYS 1
	)
add_dependencies(ext_ompl_app ext_fcl)

SET(EASY_LOCAL_SRC)
AUX_SOURCE_DIRECTORY(lib/pyse3ompl/ EASY_LOCAL_SRC)
message("PYSE3OMPL sources: ${EASY_LOCAL_SRC}")
pybind11_add_module(pyse3ompl ${EASY_LOCAL_SRC})
add_dependencies(pyse3ompl ext_ompl_app)
# add_library(ompl SHARED IMPORTED)
# set_target_properties(ompl PROPERTIES IMPORTED_LOCATION ${EXTERNAL_PROJECTS_INSTALL_PREFIX})
# add_library(ompl_app_base SHARED IMPORTED)
# set_target_properties(ompl_app_base PROPERTIES IMPORTED_LOCATION ${EXTERNAL_PROJECTS_INSTALL_PREFIX})
target_include_directories(pyse3ompl BEFORE PRIVATE ${PYTHON_INCLUDE_DIR} ${EXTERNAL_PROJECTS_INSTALL_PREFIX}/include)
target_link_directories(pyse3ompl BEFORE PRIVATE ${EXTERNAL_PROJECTS_INSTALL_PREFIX}/lib ${EXTERNAL_PROJECTS_INSTALL_PREFIX}/lib64)
target_link_libraries(pyse3ompl PRIVATE ompl ompl_app_base)
FINALIZE(pyse3ompl)

ExternalProject_Add(ext_libgeokey
	PREFIX ${EXTERNAL_PROJECTS_PREFIX}
	SOURCE_DIR ${CMAKE_SOURCE_DIR}/third-party/libgeokey/
	CMAKE_ARGS -DBUILD_VISUALIZER=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${EXTERNAL_PROJECTS_INSTALL_PREFIX}
	BUILD_COMMAND make -j ${NCPU}
	BUILD_ALWAYS 1)

SET(EASY_LOCAL_SRC)
AUX_SOURCE_DIRECTORY(lib/pygeokey/ EASY_LOCAL_SRC)
pybind11_add_module(pygeokey ${EASY_LOCAL_SRC})
add_dependencies(pygeokey ext_libgeokey)
target_include_directories(pygeokey BEFORE PRIVATE ${PYTHON_INCLUDE_DIR} ${EXTERNAL_PROJECTS_INSTALL_PREFIX}/include)
target_link_directories(pygeokey BEFORE PRIVATE ${EXTERNAL_PROJECTS_INSTALL_PREFIX}/lib ${EXTERNAL_PROJECTS_INSTALL_PREFIX}/lib64)
target_link_libraries(pygeokey PRIVATE geokey)
FINALIZE(pygeokey)

add_custom_command(TARGET pyosr POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_SOURCE_DIR}/src/GP/facade.py ${CMAKE_SOURCE_DIR}/bin/facade.py
)
add_custom_command(TARGET pyse3ompl POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_SOURCE_DIR}/src/GP/se3solver.py ${CMAKE_SOURCE_DIR}/bin/se3solver.py
)
add_custom_command(TARGET pyse3ompl POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_SOURCE_DIR}/src/GP/pds_edge.py ${CMAKE_SOURCE_DIR}/bin/pds_edge.py
)
add_custom_command(TARGET pyse3ompl POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_SOURCE_DIR}/src/GP/forest_dijkstra.py ${CMAKE_SOURCE_DIR}/bin/forest_dijkstra.py
)

# vim: tw=0
