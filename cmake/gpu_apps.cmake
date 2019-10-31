EASYLIB(ccdxx)
EASYLIB(omplaux ccdxx)

#
# All visualizer require GPU
#

EASYLIB(renderer ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES} ${GLFW3_LIBRARY})
SET(viscommon renderer ${VISUAL_PACK})

EASYADD(visheat ${CMAKE_THREAD_LIBS_INIT} ${VISUAL_PACK} tetio heatio advplyio)
EASYADD(freecube omplaux ${CMAKE_THREAD_LIBS_INIT} ${viscommon} ccdxx)
use_fcl(freecube)

SET(octbuildercommon goct vecio omplaux ${CMAKE_THREAD_LIBS_INIT})

EASYADD(octbuilder ${octbuildercommon} ${viscommon} ccdxx)
use_fcl(octbuilder)

EASYADD(naivebuilder ${octbuildercommon} ${viscommon})
use_fcl(naivebuilder)

EASYADD(tbuilder)
target_link_libraries(tbuilder PRIVATE ${octbuildercommon} ${viscommon} erocol)
use_fcl(tbuilder)
