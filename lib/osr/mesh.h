#ifndef OSR_MESH_H
#define OSR_MESH_H

#include <GL/glew.h>
#include <glm/ext.hpp>
#include <string>
#include <vector>
#include <memory>

#include <assimp/scene.h>
#include <assimp/Importer.hpp>

#include "bbox.h"
#include "camera.h"
#include "geometry.h"
#include "quickgl.h"

namespace osr {
class Scene;
class CDModel;

class Mesh {
	friend class Scene;
	std::vector<Vertex> vertices_;
	std::vector<uint32_t> indices_;
	GLuint vao_;
	GLuint vbo_;
	GLuint ibo_;
	std::shared_ptr<Mesh> shared_from_;
	bool empty_mesh_ = false;
public:
	Mesh(std::shared_ptr<Mesh> other);
	Mesh(aiMesh* mesh, glm::vec3 color);
	virtual ~Mesh();

	void render(GLuint program, Camera& camera, glm::mat4 globalXform);
	std::vector<Vertex>& getVertices();
	std::vector<uint32_t>& getIndices();

	size_t getNumberOfFaces() const;
	void addToCDModel(const glm::mat4&, CDModel&) const;
private:
	void init();
};
}

#endif /* end of include guard: MESH_H */
