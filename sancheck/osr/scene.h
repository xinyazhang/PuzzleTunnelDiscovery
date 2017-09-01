#ifndef SCENE_H
#define SCENE_H

#include <string>
#include <vector>

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/ext.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "mesh.h"
#include "camera.h"
#include "node.h"

class Scene
{
    size_t              numVertices;
    BoundingBox         bbox;
    glm::vec3           center;
    glm::mat4           xform;
    Node*               root = nullptr;
    std::vector<Mesh*>  meshes;
public:
    Scene();
    Scene(Scene& rhs);
    virtual ~Scene();

    void load(std::string filename);
    void render(GLuint program, Camera& camera, glm::mat4 globalXform);
    void clear();

    glm::mat4 transform() const { return xform; }
    BoundingBox getBoundingBox() const { return bbox; }

    void moveToCenter() {
        translate(-center);
    }
    void resetTransform() {
        xform = glm::mat4();
    }
    void translate(glm::vec3 offset) {
        xform = glm::translate(xform, offset);
    }
    void translate(float x, float y, float z) {
        xform = glm::translate(xform, glm::vec3(x, y, z));
    }
    void scale(glm::vec3 factor) {
        xform = glm::scale(xform, factor);
    }
    void scale(float x, float y, float z) {
        xform = glm::scale(xform, glm::vec3(x, y, z));
    }
    void rotate(float rad, glm::vec3 axis) {
        xform = glm::rotate(xform, rad, axis);
    }
    void rotate(float rad, float x, float y, float z) {
        xform = glm::rotate(xform, rad, glm::vec3(x, y, z));
    }

    std::vector<Mesh*>  getMeshes() { return meshes; }
private:
    void updateBoundingBox(Node* node, glm::mat4 m);
    void render(GLuint program, Camera& camera, glm::mat4 m, Node* node);
};

#endif /* end of include guard: SCENE_H */
