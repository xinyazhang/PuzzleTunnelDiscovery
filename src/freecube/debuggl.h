#ifndef DEBUGGL_H
#define DEBUGGL_H

void debugglTerminate();

#define CHECK_SUCCESS(x)   \
  do {                     \
    if (!(x)) {            \
      debugglTerminate();     \
      exit(EXIT_FAILURE);  \
    }                      \
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
      std::cerr << __func__ << " Line :" << __LINE__                                     \
                << " Status: " << status                                    \
                << " OpenGL Shader Error: Log = \n"                         \
                << &log[0];                                                 \
      std::cerr << length << " bytes\n";                                    \
      debugglTerminate();                                                      \
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
      debugglTerminate();                                                       \
      exit(EXIT_FAILURE);                                                    \
    }                                                                        \
  } while (0)

#define CHECK_GL_ERROR(statement)                                             \
  do {                                                                        \
    { statement; }                                                            \
    GLenum error = GL_NO_ERROR;                                               \
    if ((error = glGetError()) != GL_NO_ERROR) {                              \
      std::cerr << __func__ << " Line :" << __LINE__ << " OpenGL Error: code  = " << error \
                << " description =  " << DebugGLErrorToString(int(error));    \
      debugglTerminate();                                                        \
      exit(EXIT_FAILURE);                                                     \
    }                                                                         \
  } while (0)

const char* DebugGLErrorToString(int error);

#endif
