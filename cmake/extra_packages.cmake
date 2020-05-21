find_package(PkgConfig REQUIRED)

# Mult-threads

find_package(Threads REQUIRED)
SET(CMAKE_THREAD_LIBS_INIT Threads::Threads)
MESSAGE("Threads: ${CMAKE_THREAD_LIBS_INIT}")
message(STATUS "AFTER Thread CMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}")
find_package(OpenMP REQUIRED)

# GPU and visualizer
SET(USE_GPU FALSE)

SET(OpenGL_GL_PREFERENCE GLVND) # Prefer GLVND since all our platforms are using it
find_package(OpenGL QUIET)
if (OpenGL_FOUND)
	add_library(glad STATIC ${CMAKE_SOURCE_DIR}/third-party/glad/src/glad.c)
	target_include_directories(glad
		PUBLIC ${CMAKE_SOURCE_DIR}/third-party/glad/include
		)
	target_include_directories(glad
		INTERFACE ${CMAKE_SOURCE_DIR}/third-party/glad/include
		)
	set_property(TARGET glad PROPERTY POSITION_INDEPENDENT_CODE ON)
	set(OpenGL_Loader_Target glad)
endif()
pkg_search_module(GLFW3 QUIET glfw3)

if (OpenGL_FOUND AND GLFW3_FOUND)
	SET(USE_GPU TRUE)
	SET(VISUAL_PACK ${OPENGL_LIBRARIES} ${OpenGL_Loader_Target} ${GLFW3_LIBRARIES})
	message("VISUAL_PACK ${VISUAL_PACK}")
	
	message(STATUS "Using GPU ${USE_GPU}")
else ()
	message(WARNING "GPU related functions are NOT enabled")
endif ()

# Apple
if (APPLE)
	find_library(COCOA_LIBRARY Cocoa REQUIRED)
endif(APPLE)

# Double precision ODE, then single
pkg_search_module(ODE QUIET ode-double)
if (NOT ODE_FOUND)
	pkg_search_module(ODE REQUIRED ode)
endif (NOT ODE_FOUND)
message("ODE_INCLUDE_DIRS ${ODE_INCLUDE_DIRS}")
message("ODE_CFLAGS ${ODE_CFLAGS}")

# Include lib/ so every app under src/ can use them
include_directories(${CMAKE_SOURCE_DIR}/lib/)

# Include Eigen3
find_package(Eigen3 QUIET)
include_directories(SYSTEM ${EIGEN3_INCLUDE_DIR})

# include_directories(${IGL_INCLUDE_DIR})

find_package(Boost 1.54.0 REQUIRED COMPONENTS system thread)
MESSAGE("Boost include dir: ${Boost_INCLUDE_DIR}")
MESSAGE("Boost libraries: ${Boost_LIBRARIES}")
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIR})
message(STATUS "AFTER Boost CMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}")

# CGAL Mess
# Can be disabled
option(MKOBS_USE_CGAL "Enable functions that depend on CGAL" ON)
option(MKOBS_FORCE_CGAL "Make CGAL REQUIRED instead of optional" ON)
if (MKOBS_USE_CGAL)
	if (MKOBS_FORCE_CGAL)
		find_package(CGAL REQUIRED COMPONENTS Core)
	else (MKOBS_FORCE_CGAL)
		find_package(CGAL QUIET COMPONENTS Core)
	endif (MKOBS_FORCE_CGAL)
	message(STATUS "AFTER CGAL CMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}")
	#SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CGAL_MODULES_DIR}")
endif (MKOBS_USE_CGAL)

# set(CGAL_LIB "${CGAL_LIB};CGAL")
# message(STATUS "CGAL USE FILE ${CGAL_USE_FILE}")
# message(STATUS "CGAL LIBRARIES ${CGAL_LIBRARIES}")
# message(STATUS "CGAL LIBRARY ${CGAL_LIBRARY}")
# set(CGAL_LIB "${CGAL_LIBRARY};${CGAL_Core_LIBRARY}")
# message(STATUS "CGAL LIB ${CGAL_LIB}")
message(STATUS "CGAL_FOUND: ${CGAL_FOUND}")

find_package(GMP QUIET)
message(STATUS "AFTER GMP CMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}")
find_package(MPFR QUIET)
message(STATUS "AFTER MPFR CMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}")
find_package(SUITESPARSE QUIET)

if (${SUITESPARSE_FOUND})
#LINK_DIRECTORIES(/usr/lib64/openmpi/lib/)
	EASYADD(heat)
	target_include_directories(heat ${SUITESPARSE_INCLUDE_DIRS})
#TARGET_LINK_LIBRARIES(heat ${SUITESPARSE_LIBRARIES} vecio mkl_core mkl_intel_thread mkl_intel_lp64 iomp5 ptscotch scotch scotcherr pastix mpi z)
#TARGET_LINK_LIBRARIES(heat ${SUITESPARSE_LIBRARIES} vecio mkl_core mkl_intel_thread mkl_intel_lp64 iomp5)
	message(STATUS "SUITESPARSE LIBS: ${SUITESPARSE_LIBRARIES}")
	target_link_libraries(heat ${SUITESPARSE_LIBRARIES} vecio)
#TARGET_INCLUDE_DIRECTORIES(heat BEFORE PRIVATE /usr/include/openmpi-x86_64/)
#TARGET_INCLUDE_DIRECTORIES(heat BEFORE PRIVATE /usr/local/include/pastix/)
#TARGET_INCLUDE_DIRECTORIES(heat BEFORE PRIVATE ${CMAKE_SOURCE_DIR}/third-party/eigen)
	SANCHECK(laplacian)
	target_link_libraries(sancheck_laplacian tetio)
else ()
	message("SuiteSparse is not found")
endif ()

find_package(vhacd QUIET)
find_package(OpenVDB QUIET)

message("OpenVDB_FOUND: ${OpenVDB_FOUND}")

pkg_search_module(CCD REQUIRED ccd)
message("CCD_LIBRARIES ${CCD_LIBRARIES}")
message("CCD_LIBRARY_DIRS ${CCD_LIBRARY_DIRS}")

function(use_ccd TGT)
	target_link_directories(${TGT} PRIVATE ${CCD_LIBRARY_DIRS})
	target_link_libraries(${TGT} PRIVATE ${CCD_LIBRARIES})
endfunction(use_ccd)
# include_directories(${CCD_INCLUDE_DIRS})

find_package(PNG REQUIRED) # Used by OSR

find_package(Python3 COMPONENTS Development)
# find_package(pybind11 REQUIRED)
add_subdirectory(third-party/pybind11/)
message(STATUS "Python version string: ${PYTHONLIBS_VERSION_STRING}")
