#ifndef GEOMETRY_H
#define GEOMETRY_H

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

struct Vertex {
    glm::vec3 position;
    glm::vec3 color;
};

#endif /* end of include guard: GEOMETRY_H */
