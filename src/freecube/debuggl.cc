#include "debuggl.h"
#include <GL/gl.h>
#include <GLFW/glfw3.h>

const char* DebugGLErrorToString(int error) {
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

void debugglTerminate()
{
	glfwTerminate();
}
