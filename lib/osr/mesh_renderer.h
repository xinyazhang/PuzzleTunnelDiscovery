#ifndef OFF_SCREEN_RENDERING_MESH_RENDERER_H
#define OFF_SCREEN_RENDERING_MESH_RENDERER_H

#if GPU_ENABLED

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <GL/glew.h>
#include <memory>

namespace osr {

class Camera;
class Mesh;
using std::shared_ptr;

/*
 * MeshRenderer
 * 
 *      Render a mesh object to rendering target.
 *      
 *      This class also maintains corresponding GL objects.
 *      
 *      Note: do not create MeshRenderer for empty Mesh, otherwise the
 *            behavior is undefined.
 * 
 */
class MeshRenderer {
private:
	GLuint vao_;
	GLuint vbo_;
	GLuint ibo_;
	size_t number_of_faces_;
	shared_ptr<MeshRenderer> shared_from_;

	void initVAO();
	void uploadData(const shared_ptr<Mesh>& mesh);
	void bindAttributes();
public:
	MeshRenderer(shared_ptr<Mesh> mesh);
	MeshRenderer(shared_ptr<MeshRenderer> other);
	~MeshRenderer();
	
	void render(GLuint program, Camera& camera, glm::mat4 globalXform);
};

}

#endif // GPU_ENABLED

#endif
