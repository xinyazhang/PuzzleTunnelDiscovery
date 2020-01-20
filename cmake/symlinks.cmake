add_custom_command(TARGET pyosr POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_SOURCE_DIR}/src/GP/facade.py ${CMAKE_SOURCE_DIR}/bin/facade.py
)
add_custom_command(TARGET pyse3ompl POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_SOURCE_DIR}/src/GP/se3solver.py ${CMAKE_SOURCE_DIR}/bin/se3solver.py
)
add_custom_command(TARGET pyse3ompl POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_SOURCE_DIR}/src/GP/SolVis.py ${CMAKE_SOURCE_DIR}/bin/SolVis.py
)
add_custom_command(TARGET pyse3ompl POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_SOURCE_DIR}/src/GP/BTexVis.py ${CMAKE_SOURCE_DIR}/bin/BTexVis.py
)
add_custom_command(TARGET pyse3ompl POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_SOURCE_DIR}/src/GP/pds_edge.py ${CMAKE_SOURCE_DIR}/bin/pds_edge.py
)
add_custom_command(TARGET pyse3ompl POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_SOURCE_DIR}/src/GP/forest_dijkstra.py ${CMAKE_SOURCE_DIR}/bin/forest_dijkstra.py
)
