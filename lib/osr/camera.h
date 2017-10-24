#ifndef OSR_CAMERA_H
#define OSR_CAMERA_H

#include <stack>
#include <iostream>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "quickgl.h"
#include "uniform.h"

namespace osr {
class Camera {
    UniformMatrix           matrix;
    std::stack<glm::mat4>   stack;
public:
    Camera();
    Camera(const Camera& rhs);
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

    void uniform(GLuint program, glm::mat4 xform);
};
}

#endif /* end of include guard: CAMERA_H */
