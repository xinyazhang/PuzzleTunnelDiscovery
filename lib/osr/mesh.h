#ifndef MESH_H
#define MESH_H

#include <GL/glew.h>
#include <glm/ext.hpp>
#include <string>
#include <vector>

#include <assimp/scene.h>
#include <assimp/Importer.hpp>

#include "bbox.h"
#include "camera.h"
#include "geometry.h"
#include "quickgl.h"

namespace osr {
class Scene;

class Mesh {
	friend class Scene;
	std::vector<Vertex> vertices;
	std::vector<uint32_t> indices;
	GLuint vao;
	GLuint vbo;
	GLuint ibo;

public:
	Mesh(Mesh& rhs);
	Mesh(aiMesh* mesh, glm::vec3 color);
	virtual ~Mesh();

	void render(GLuint program, Camera& camera, glm::mat4 globalXform);
	std::vector<Vertex>& getVertices() { return vertices; };
	std::vector<uint32_t>& getIndices() { return indices; };

private:
	void init();
};
}

#endif /* end of include guard: MESH_H */
