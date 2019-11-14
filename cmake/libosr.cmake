EASYLIB(osr assimp ${Boost_LIBRARIES} tritri ${PNG_LIBRARIES})

## Link common libraries
use_ccd(osr)
use_ompl(osr)
use_fcl(osr)

target_compile_definitions(osr PRIVATE -DGLM_ENABLE_EXPERIMENTAL -DGLM_FORCE_SIZE_FUNC=1 -DGLM_FORCE_RADIANS=1)


## Link osr to ode
target_include_directories(osr BEFORE PRIVATE ${ODE_INCLUDE_DIRS})
target_compile_definitions(osr PRIVATE ${ODE_CFLAGS}) # Required, float or double
target_link_libraries(osr PRIVATE ${ODE_LIBRARIES})

## OPTIONAL FUNCTION 1: rendering
if (USE_GPU)
	target_compile_definitions(osr PUBLIC GPU_ENABLED=1)
	target_link_libraries(osr PRIVATE gbm EGL ${OPENGL_LIBRARIES} ${Boost_LIBRARIES})
	target_link_libraries(osr PUBLIC ${OpenGL_Loader_Target})

	EASYADD(vistexture osr ${VISUAL_PACK})
	# target_compile_definitions(vistexture PRIVATE GPU_ENABLED=1)

	EASYADD(cfreeman osr ${VISUAL_PACK})
	# target_compile_definitions(cfreeman PRIVATE GPU_ENABLED=1)

	EASYADD(vispath osr ${VISUAL_PACK})
	# target_compile_definitions(vispath PRIVATE GPU_ENABLED=1)

else (USE_GPU)
	target_compile_definitions(osr PUBLIC GPU_ENABLED=0)
endif (USE_GPU)

## OPTIONAL FUNCTION 2: meshbool
if (TARGET meshbool) # Only link if meshbool exists
	target_link_libraries(osr PRIVATE meshbool)
	target_compile_definitions(osr PUBLIC PYOSR_HAS_MESHBOOL=1)
else ()
	target_compile_definitions(osr PUBLIC PYOSR_HAS_MESHBOOL=0)
endif()
