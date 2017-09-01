#ifndef UNIFORM_H
#define UNIFORM_H

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

struct UniformMatrix {
    glm::mat4 model;
    glm::mat4 view;
    glm::mat4 proj;

    UniformMatrix() {
        model = glm::mat4();
        view = glm::mat4();
        proj = glm::mat4();
    }

    UniformMatrix(UniformMatrix& rhs) {
        model   = rhs.model;
        view    = rhs.view;
        proj    = rhs.proj;
    }
};


#endif /* end of include guard: UNIFORM_H */
