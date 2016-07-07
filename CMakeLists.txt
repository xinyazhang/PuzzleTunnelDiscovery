cmake_minimum_required(VERSION 3.0)

project(mkobs3d)
set (mkobs3d_VERSION_MAJOR 0)
set (mkobj3d_VERSION_MINOR 2)

# Include Eigen3 and IGL
FIND_PACKAGE(Eigen3 REQUIRED)
INCLUDE_DIRECTORIES(${EIGEN3_INCLUDE_DIR})
INCLUDE_DIRECTORIES(${IGL_INCLUDE_DIR})

# C++11 and Debug
# Note: we focus on gcc/clang right now, so we uses compiler specific solutions.
set(CMAKE_CXX_FLAGS "--std=c++14 -g -ggdb3 -Wno-ignored-attributes -Wno-deprecated-declarations -Wno-deprecated ${CMAKE_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "-g")

# Define function to copy binaries to bin/
FUNCTION(FINALIZE target files)
	add_custom_command(TARGET ${target} POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E copy ${files} ${CMAKE_SOURCE_DIR}/bin
	)
ENDFUNCTION()

FUNCTION(EASYADD place)
	AUX_SOURCE_DIRECTORY(src/${place}/ EASY_LOCAL_SRC)
	ADD_EXECUTABLE(${place} ${EASY_LOCAL_SRC})
	FINALIZE(${place} "${CMAKE_BINARY_DIR}/${place}")
ENDFUNCTION()

#THIRD PARTIES
# OBJ2PLY
include(ExternalProject)
SET(OBJ2PLY_BIN "${CMAKE_BINARY_DIR}/bin")
SET(OBJ2PLY_ARGS "-DIGL_INCLUDE_DIR=${IGL_INCLUDE_DIR}")
LIST(APPEND OBJ2PLY_ARGS "-DEIGEN3_INCLUDE_DIR=${EIGEN3_INCLUDE_DIR}")
ExternalProject_Add(OBJ2PLY
	GIT_REPOSITORY "https://github.com/xinyazhang/obj2ply.git"
	GIT_SUBMODULES "${CMAKE_SOURCE_DIR}/third-party/obj2ply/"
	CMAKE_ARGS ${OBJ2PLY_ARGS}
	INSTALL_COMMAND ""
	BINARY_DIR "${OBJ2PLY_BIN}")
FINALIZE(OBJ2PLY ${OBJ2PLY_BIN}/obj2ply)


# Main program
SET(BOOLEAN_SRC src/boolean/join.cc src/boolean/join.h)
ADD_EXECUTABLE(blend ${BOOLEAN_SRC} src/boolean/main.cc)
ADD_EXECUTABLE(subtract ${BOOLEAN_SRC} src/boolean/substract.cc)
FINALIZE(blend "${CMAKE_BINARY_DIR}/blend")
FINALIZE(subtract "${CMAKE_BINARY_DIR}/subtract")

find_package(Threads REQUIRED)
find_package(CGAL COMPONENTS Core REQUIRED)
include(${CGAL_USE_FILE})
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CGAL_MODULES_DIR}")
find_package(GMP REQUIRED)
find_package(MPFR REQUIRED)

SET(BOOLEAN_LIB ${CGAL_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${GMP_LIBRARIES} ${MPFR_LIBRARIES})

TARGET_LINK_LIBRARIES(blend ${BOOLEAN_LIB})
TARGET_LINK_LIBRARIES(subtract ${BOOLEAN_LIB})

EASYADD(meshgen)
EASYADD(mkgen)

#FILE(COPY ${BINARIES} DESTINATION ${CMAKE_SOURCE_DIR}/bin)

# vim: tw=0
