#if GPU_ENABLED

#include "camera.h"
#include <glm/gtx/io.hpp>

namespace osr {
Camera::Camera() {
}

Camera::Camera(const Camera& rhs) {
    matrix = rhs.matrix;
    stack  = rhs.stack;
}

Camera::~Camera() {

}

void
Camera::pushMatrix() {
    stack.push(matrix.model);
}

void
Camera::popMatrix() {
    matrix.model = stack.top();
    stack.pop();
}

void
Camera::translateX(float v) {
    glm::vec3 offset = glm::vec3(v, 0, 0);
    translate(offset);
}

void
Camera::translateY(float v) {
    glm::vec3 offset = glm::vec3(0, v, 0);
    translate(offset);
}

void
Camera::translateZ(float v) {
    glm::vec3 offset = glm::vec3(0, 0, v);
    translate(offset);
}

void
Camera::translate(float x, float y, float z) {
    glm::vec3 offset = glm::vec3(x, y, z);
    translate(offset);
}

void
Camera::translate(glm::vec3 offset) {
    matrix.model = glm::translate(matrix.model, offset);
}

void
Camera::scaleX(float v) {
    glm::vec3 factor = glm::vec3(v, 0, 0);
    scale(factor);
}

void
Camera::scaleY(float v) {
    glm::vec3 factor = glm::vec3(0, v, 0);
    scale(factor);
}

void
Camera::scaleZ(float v) {
    glm::vec3 factor = glm::vec3(0, 0, v);
    scale(factor);
}

void
Camera::scale(float x, float y, float z) {
    glm::vec3 factor = glm::vec3(x, y, z);
    scale(factor);
}

void
Camera::scale(glm::vec3 factor) {
    matrix.model = glm::scale(matrix.model, factor);
}

void
Camera::rotateX(float rad, float v) {
    glm::vec3 axis = glm::vec3(v, 0, 0);
    rotate(rad, axis);
}

void
Camera::rotateY(float rad, float v) {
    glm::vec3 axis = glm::vec3(0, v, 0);
    rotate(rad, axis);
}

void
Camera::rotateZ(float rad, float v) {
    glm::vec3 axis = glm::vec3(0, 0, v);
    rotate(rad, axis);
}

void
Camera::rotate(float rad, float x, float y, float z) {
    glm::vec3 axis = glm::vec3(x, y, z);
    rotate(rad, axis);
}

void
Camera::rotate(float rad, glm::vec3 axis) {
    matrix.model = glm::rotate(matrix.model, rad, axis);
}

void
Camera::lookAt(glm::vec3 eye, glm::vec3 center, glm::vec3 up) {
    matrix.view = glm::lookAt(eye, center, up);
}

void
Camera::ortho2D(float left, float right, float bottom, float top) {
    matrix.proj = glm::ortho(left, right, bottom, top);
}

void
Camera::ortho(float left, float right, float bottom, float top, float near, float far) {
    matrix.proj = glm::ortho(left, right, bottom, top, near, far);
}

void
Camera::frustum(float left, float right, float bottom, float top, float near, float far) {
    matrix.proj = glm::frustum(left, right, bottom, top, near, far);
}

void
Camera::perspective(float fovy, float aspectRatio, float near, float far) {
    matrix.proj = glm::perspective(fovy, aspectRatio, near, far);
}

void
Camera::uniform(GLuint program, glm::mat4 xform)
{
	GLint modelLoc, viewLoc, projLoc;
	CHECK_GL_ERROR(modelLoc = glGetUniformLocation(program, "model"));
	CHECK_GL_ERROR(viewLoc  = glGetUniformLocation(program, "view"));
	CHECK_GL_ERROR(projLoc  = glGetUniformLocation(program, "proj"));
	if (modelLoc == -1) { std::cerr << "cannot find model matrix uniform location" << std::endl; exit(EXIT_FAILURE); }
	if (viewLoc == -1)  { std::cerr << "cannot find view matrix uniform location"  << std::endl; exit(EXIT_FAILURE); }
	if (projLoc == -1)  { std::cerr << "cannot find proj matrix uniform location"  << std::endl; exit(EXIT_FAILURE); }
	CHECK_GL_ERROR(glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(matrix.model * xform)));
	// std::cerr << "Render with\n" << matrix.model * xform << "\n";
	CHECK_GL_ERROR(glUniformMatrix4fv(viewLoc,  1, GL_FALSE, glm::value_ptr(matrix.view)));
	CHECK_GL_ERROR(glUniformMatrix4fv(projLoc,  1, GL_FALSE, glm::value_ptr(matrix.proj)));
}

}

#endif // GPU_ENABLED
