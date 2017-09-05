#ifndef QUICKGL_H
#define QUICKGL_H

#ifndef QUICKGL_NOT_INCLUDE_GLEW
#include <GL/glew.h>
#endif

#define CHECK_SUCCESS(x)   \
  do {                     \
    if (!(x)) {            \
      exit(EXIT_FAILURE);  \
    }                      \
  } while (0)

#define CHECK_GL_ERROR(statement)                                             \
  do {                                                                        \
    { statement; }                                                            \
    GLenum error = GL_NO_ERROR;                                               \
    if ((error = glGetError()) != GL_NO_ERROR) {                              \
      std::cerr                                                               \
                << "File: " << __FILE__ << std::endl                          \
                << "Func: " << __func__ << std::endl                          \
                << "Line: " << __LINE__ << std::endl                          \
                << "Erno: " << error << std::endl                             \
                << "Desc: " << DebugGLErrorToString(int(error))               \
                << std::endl;                                                 \
      exit(EXIT_FAILURE);                                                     \
    }                                                                         \
  } while (0)

const char* DebugGLErrorToString(int error);

void CheckShaderCompilation(GLuint shader);
void CheckProgramLinkage(GLuint program);

#endif
