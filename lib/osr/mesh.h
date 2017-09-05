#ifndef MESH_H
#define MESH_H

#include <GL/glew.h>
#include <string>
#include <vector>
#include <glm/ext.hpp>

#include <assimp/Importer.hpp>
#include <assimp/scene.h>

#include "geometry.h"
#include "bbox.h"
#include "quickgl.h"
#include "camera.h"

namespace osr {
class Scene;

class Mesh {
    friend class Scene;
    std::vector<Vertex>     vertices;
    std::vector<uint32_t>   indices;
    GLuint                  vao;
    GLuint                  vbo;
    GLuint                  ibo;
public:
    Mesh(Mesh& rhs);
    Mesh(aiMesh* mesh, glm::vec3 color);
    virtual ~Mesh();

    void render(GLuint program, Camera& camera, glm::mat4 globalXform);
    std::vector<Vertex>  &   getVertices() { return vertices; };
    std::vector<uint32_t>&   getIndices() { return indices; };
private:
    void init();
};
}

#endif /* end of include guard: MESH_H */
