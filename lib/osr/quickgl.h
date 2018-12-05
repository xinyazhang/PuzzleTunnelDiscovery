#ifndef OSR_QUICKGL_H
#define OSR_QUICKGL_H

#if GPU_ENABLED

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

#define CHECK_GL_SHADER_ERROR(id)                                           \
  do {                                                                      \
    GLint status = 0;                                                       \
    GLint length = 0;                                                       \
    glGetShaderiv(id, GL_COMPILE_STATUS, &status);                          \
    glGetShaderiv(id, GL_INFO_LOG_LENGTH, &length);                         \
    if (status != GL_TRUE) {                                                \
      std::string log(length, 0);                                           \
      glGetShaderInfoLog(id, length, nullptr, &log[0]);                     \
      std::cerr << "Function: " << __func__                                 \
	        << " Line :" << __LINE__                                    \
	        << " FILE :" << __FILE__                                    \
                << " Status: " << status                                    \
                << " OpenGL Shader Error: Log = \n"                         \
                << &log[0];                                                 \
      std::cerr << length << " bytes\n";                                    \
      exit(EXIT_FAILURE);                                                   \
    }                                                                       \
  } while (0)

#define CHECK_GL_PROGRAM_ERROR(id)                                           \
  do {                                                                       \
    GLint status = 0;                                                        \
    GLint length = 0;                                                        \
    glGetProgramiv(id, GL_LINK_STATUS, &status);                             \
    glGetProgramiv(id, GL_INFO_LOG_LENGTH, &length);                         \
    if (status != GL_TRUE) {                                                 \
      std::string log(length, 0);                                            \
      glGetProgramInfoLog(id, length, nullptr, &log[0]);                     \
      std::cerr << __func__ << " Line :" << __LINE__ << " OpenGL Program Error: Log = \n" \
                << &log[0];                                                  \
      exit(EXIT_FAILURE);                                                    \
    }                                                                        \
  } while (0)

const char* DebugGLErrorToString(int error);

void CheckShaderCompilation(GLuint shader);
void CheckProgramLinkage(GLuint program);

#endif // GPU_ENABLED

#endif
