if (APPLE)
  link_directories(
    "/usr/local/lib"
    "/opt/local/lib"
  )
endif()

include_directories(
  "/usr/local/include"
  "/opt/local/include"
  "/usr/local/Cellar/glew/2.0.0/include/"
  "/Users/randallsmith/Documents/sdks/git/libigl/include"
  "/lusr/include")
link_directories("$ENV{HOME}/.local/lib")

