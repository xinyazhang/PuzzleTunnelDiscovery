PYADD(pyosr osr)
use_fcl(pyosr)

PYADD(pyse3ompl ompl ompl_app_base)
add_dependencies(pyse3ompl ext_ompl_app)
use_fcl(pyse3ompl)

PYADD(pygeokey geokey)
add_dependencies(pygeokey ext_libgeokey)

if (USE_GPU)
PYADD(pyvistexture osr ${VISUAL_PACK})
add_dependencies(pyvistexture osr)
endif()

PYADD(pycutec2)
