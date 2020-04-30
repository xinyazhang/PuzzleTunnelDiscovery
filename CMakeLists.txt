cmake_minimum_required(VERSION 3.13)
# We need 3.13+ to support target_include_directories
# Alternative solutions (e.g. set_target_properties) cannot override the system's ompl library

project(mkobs3d CXX C)
# SET(CMAKE_BUILD_TYPE Debug) # Seems CMAKE does not recognize -DCMAKE_BUILD_TYPE
set (mkobs3d_VERSION_MAJOR 0)
set (mkobs3d_VERSION_MINOR 9)

include(ProcessorCount)
ProcessorCount(NCPU)

# C++14 and Debug
message(STATUS "BEFORE C++14 CMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${OpenMP_CXX_FLAGS} ${CMAKE_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "-g")
message(STATUS "AFTER C++14 CMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}")

message(STATUS "Initial CMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}")
message(STATUS "Initial Python links to: ${PYTHON_LIBRARY}")

include(cmake/bundles.cmake)
include(cmake/extra_directories.cmake)
include(cmake/extra_packages.cmake)
include(cmake/extra_functions.cmake)

# SET(CMAKE_BUILD_TYPE Debug)

IF (CGAL_FOUND)
	message(STATUS "Find CGAL at\n\tHeaders:\t${CGAL_INCLUDE_DIRS}\n\tLibs: ${CGAL_LIB}")
	EASYLIB(meshbool)
	target_include_directories(meshbool BEFORE PRIVATE ${CGAL_INCLUDE_DIRS})
	target_link_libraries(meshbool CGAL::CGAL CGAL::CGAL_Core ${CMAKE_THREAD_LIBS_INIT} ${GMP_LIBRARIES} ${MPFR_LIBRARIES} ${Boost_LIBRARIES})
ELSE (CGAL_FOUND)
	message(WARNING "CGAL NOT FOUND, Some functions and libraires will not be built")
ENDIF (CGAL_FOUND)

IF (TARGET meshbool)
	# Main program
	add_executable(blend src/boolean/main.cc)
	add_executable(subtract src/boolean/substract.cc)
	FINALIZE(blend)
	FINALIZE(subtract)
	SET(BOOLEAN_LIB meshbool)
	target_link_libraries(blend ${BOOLEAN_LIB})
	target_link_libraries(subtract ${BOOLEAN_LIB})

	FILTERADD(addcover)
	FILTERADD(hollow)
	target_link_libraries(addcover meshbool)
	target_link_libraries(hollow meshbool)
ENDIF (TARGET meshbool)

EASYLIB(tetio)
EASYLIB(advplyio)
EASYLIB(heatio)
EASYLIB(vecio)
EASYLIB(geopick)
EASYLIB(mazeinfo advplyio)
EASYLIB(goct)
EASYLIB(tritri)
EASYLIB(omplaux tetio)
use_fcl(omplaux)

EASYADD(maze2to3)
EASYADD(meshgen mazeinfo)
EASYADD(omplgen mazeinfo)
EASYADD(mkgen)
EASYADD(dlap ${CMAKE_THREAD_LIBS_INIT} ${Boost_LIBRARIES} tetio)
EASYADD(Bcond tetio)
EASYADD(NBcond tetio vecio)
EASYADD(ring1picker tetio)
EASYADD(mass tetio vecio)
EASYADD(follow tetio heatio)
EASYADD(invgen)
EASYADD(periodicalize tetio geopick)
EASYADD(tet2obj tetio geopick)
EASYADD(mergep)
EASYADD(evadiff tetio heatio)
EASYADD(trmap omplaux)
use_fcl(trmap) # TODO: remove deps on fcl
use_ccd(trmap)

FILTERADD(flipfaces)
FILTERADD(addbb)

IF ($(vhacd_FOUND))
	EASYADD(convexpp vhacd ${CMAKE_THREAD_LIBS_INIT} ${GMP_LIBRARIES} ${MPFR_LIBRARIES} ${CGAL_LIB} ${Boost_LIBRARIES})
ELSE ($(vhacd_FOUND))
	MESSAGE(STATUS "convexpp is NOT BUILT")
ENDIF ($(vhacd_FOUND))

if (${OpenVDB_FOUND})
	EASYLIB(fat)
	target_include_directories(fat PUBLIC ${OpenVDB_INCLUDE_DIR})
	target_link_directories(fat PUBLIC ${OpenVDB_LIBRARY_DIR})
	target_link_libraries(fat ${OpenVDB_LIBRARIES} tbb)

	EASYLIB(erocol fat)
	EASYADD(levelset fat)

	if (USE_GPU)
		EASYADD(tbuilder)
		target_link_libraries(tbuilder PRIVATE ${octbuildercommon} ${viscommon} erocol)
		use_fcl(tbuilder)
	endif (USE_GPU)

	target_link_libraries(omplaux erocol)
endif ()

# Off-Screen Renderer for RL (reinforcement learning)
# However if OpenGL is disabled, then the renderer part is not compiled.
include(cmake/libosr.cmake)

EASYLIB(rectpack)
use_fcl(rectpack)
EASYADD(objautouv rectpack ${CMAKE_THREAD_LIBS_INIT})

# Blender PRM and RRT for RL's GT (Ground Truth)
EASYADD(mtblender)
use_ccd(mtblender)
use_ompl(mtblender)
use_fcl(mtblender)

EASYADD(mapmerger)
use_ccd(mapmerger)
use_ompl(mapmerger)
use_fcl(mapmerger)


include(cmake/pybinds.cmake)
include(cmake/symlinks.cmake)
include(cmake/sanchecks.cmake)

# vim: tw=0
