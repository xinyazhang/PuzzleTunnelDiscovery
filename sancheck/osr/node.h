#ifndef NODE_H
#define NODE_H

#include <vector>
#include <iostream>
#include <assimp/scene.h>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "camera.h"

class Scene;

class Node
{
    friend class Scene;
    std::vector<Node*>      nodes;
    std::vector<uint32_t>   meshes;
    glm::mat4               xform;
public:
    Node(aiNode* node);
    virtual ~Node();
};

#endif /* end of include guard: NODE_H */
