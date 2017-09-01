#ifndef CAMERA_H
#define CAMERA_H

#include <stack>
#include <iostream>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "quickgl.h"
#include "uniform.h"

class Camera
{
    UniformMatrix           matrix;
    std::stack<glm::mat4>   stack;
public:
    Camera();
    Camera(Camera& rhs);
    virtual ~Camera();

    void commit();
    UniformMatrix& getUniformMatrices() { return matrix; }

    // only push and pop for model matrix
    void pushMatrix();
    void popMatrix();

    // model matrix
    void translateX(float v);
    void translateY(float v);
    void translateZ(float v);
    void translate(float x, float y, float z);
    void translate(glm::vec3 offset);

    void scaleX(float v);
    void scaleY(float v);
    void scaleZ(float v);
    void scale(float x, float y, float z);
    void scale(glm::vec3 offset);

    void rotateX(float rad, float v);
    void rotateY(float rad, float v);
    void rotateZ(float rad, float v);
    void rotate(float rad, float x, float y, float z);
    void rotate(float rad, glm::vec3 axis);

    // view matrix
    void lookAt(glm::vec3 eye, glm::vec3 center, glm::vec3 up);

    // projection matrix
    void ortho2D(float left, float right, float bottom, float top);
    void ortho(float left, float right, float bottom, float top, float near, float far);
    void frustum(float left, float right, float bottom, float top, float near, float far);
    void perspective(float fovy, float aspectRatio, float near, float far);

    void uniform(GLuint program, glm::mat4 xform) {
        GLint modelLoc, viewLoc, projLoc;
        CHECK_GL_ERROR(modelLoc = glGetUniformLocation(program, "model"));
        CHECK_GL_ERROR(viewLoc  = glGetUniformLocation(program, "view"));
        CHECK_GL_ERROR(projLoc  = glGetUniformLocation(program, "proj"));
        if (modelLoc == -1) { std::cerr << "cannot find model matrix uniform location" << std::endl; exit(EXIT_FAILURE); }
        if (viewLoc == -1)  { std::cerr << "cannot find view matrix uniform location"  << std::endl; exit(EXIT_FAILURE); }
        if (projLoc == -1)  { std::cerr << "cannot find proj matrix uniform location"  << std::endl; exit(EXIT_FAILURE); }
        CHECK_GL_ERROR(glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(matrix.model * xform)));
        CHECK_GL_ERROR(glUniformMatrix4fv(viewLoc,  1, GL_FALSE, glm::value_ptr(matrix.view)));
        CHECK_GL_ERROR(glUniformMatrix4fv(projLoc,  1, GL_FALSE, glm::value_ptr(matrix.proj)));
    }
};

#endif /* end of include guard: CAMERA_H */
