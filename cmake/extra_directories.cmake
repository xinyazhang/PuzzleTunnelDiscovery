if (APPLE)
  link_directories(
    "/usr/local/lib"
    "/opt/local/lib"
  )
endif()

include_directories(${CMAKE_SOURCE_DIR}/third-party/libigl/include/)
