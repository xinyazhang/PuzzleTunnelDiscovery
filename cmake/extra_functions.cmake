# Define function to copy binaries to bin/
function(FINALCOPY target files)
	add_custom_command(TARGET ${target} POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E copy ${files} ${CMAKE_SOURCE_DIR}/bin
	)
endfunction()
function(FINALIZE target)
	add_custom_command(TARGET ${target} POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${target}> ${CMAKE_SOURCE_DIR}/bin
	)
endfunction()

function(EASYADD place)
	aux_source_directory(src/${place}/ EASY_LOCAL_SRC)
	add_executable(${place} ${EASY_LOCAL_SRC})
	FINALIZE(${place})
	target_link_libraries(${place} PRIVATE ${ARGN})
endfunction()

function(SANCHECK place)
	aux_source_directory(sancheck/${place}/ SAN_CHECK_SRC)
	add_executable(sancheck_${place} ${SAN_CHECK_SRC})
	FINALIZE(sancheck_${place})
	target_link_libraries(sancheck_${place} PRIVATE ${ARGN})
endfunction()

function(EASYLIB place)
	aux_source_directory(lib/${place}/ EASY_LOCAL_SRC)
	add_library(${place} SHARED ${EASY_LOCAL_SRC})
	#INCLUDE_DIRECTORIES(lib/${place})
	FINALIZE(${place})
	target_link_libraries(${place} PRIVATE ${ARGN})
endfunction()

function(FILTERADD filter)
	set(FT_LOCAL_SRC src/filters/${filter}.cc)
	ADD_EXECUTABLE(${filter} ${FT_LOCAL_SRC})
	FINALIZE(${filter})
endfunction()

function(PYADD place)
	set(EASY_LOCAL_SRC)
	aux_source_directory(lib/${place}/ EASY_LOCAL_SRC)
	pybind11_add_module(${place} ${EASY_LOCAL_SRC})
	target_include_directories(${place} BEFORE PRIVATE ${PYTHON_INCLUDE_DIR} ${EXTERNAL_PROJECTS_INSTALL_PREFIX}/include)
	target_link_directories(${place} BEFORE PRIVATE ${EXTERNAL_PROJECTS_INSTALL_PREFIX}/lib ${EXTERNAL_PROJECTS_INSTALL_PREFIX}/lib64)
	target_link_libraries(${place} PRIVATE ${ARGN})
	FINALIZE(${place})
endfunction()
