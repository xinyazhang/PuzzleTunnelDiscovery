SANCHECK(fatten)
SANCHECK(ccd)
use_ccd(sancheck_ccd)

SANCHECK(ode)
target_link_libraries(sancheck_ode ${ODE_LIBRARIES})
target_include_directories(sancheck_ode BEFORE PRIVATE ${ODE_INCLUDE_DIRS})
target_compile_definitions(sancheck_ode PRIVATE ${ODE_CFLAGS}) # Required, float or double

SANCHECK(fclcontacts)
use_ccd(sancheck_fclcontacts)
use_fcl(sancheck_fclcontacts)

if (USE_GPU)
	SANCHECK(osr)
	target_link_libraries(sancheck_osr osr)
	# target_compile_definitions(sancheck_osr PRIVATE GPU_ENABLED=1)
endif (USE_GPU)
