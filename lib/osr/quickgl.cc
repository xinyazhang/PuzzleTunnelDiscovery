#include "quickgl.h"
#include <stdlib.h>
#include <iostream>
#include <vector>

const char* DebugGLErrorToString(int error)
{
	switch (error) {
		case GL_NO_ERROR:
			return "GL_NO_ERROR";
			break;
		case GL_INVALID_ENUM:
			return "GL_INVALID_ENUM";
			break;
		case GL_INVALID_VALUE:
			return "GL_INVALID_VALUE";
			break;
		case GL_INVALID_OPERATION:
			return "GL_INVALID_OPERATION";
			break;
		case GL_OUT_OF_MEMORY:
			return "GL_OUT_OF_MEMORY";
			break;
		default:
			return "Unknown Error";
			break;
	}
	return "Unicorns Exist";
}

void ErrorCallback(int error, const char* description)
{
	std::cerr << "EGL Error: " << description << "\n";
}

void CheckShaderCompilation(GLuint shader)
{
	GLint success;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
	if (!success) {
		GLint logSize = 0;
		glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &logSize);
		std::vector<GLchar> errorLog(logSize);
		glGetShaderInfoLog(shader, logSize, &logSize, &errorLog[0]);
		std::cerr << "shader compile log:\n" << errorLog.data() << std::endl;
		glDeleteShader(shader);
		exit(EXIT_FAILURE);
	}
}

void CheckProgramLinkage(GLuint program)
{
	GLint linked;
	glGetProgramiv(program, GL_LINK_STATUS, &linked);
	if (!linked) {
		std::cerr << "failed to link program" << std::endl;
		glDeleteProgram(program);
		exit(EXIT_FAILURE);
	}
}
