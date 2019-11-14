#ifndef OFF_SCREEN_RENDERING_MESH_RENDERER_H
#define OFF_SCREEN_RENDERING_MESH_RENDERER_H

#if GPU_ENABLED

#include <glm/glm.hpp>
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
	unsigned int vao_;
	unsigned int vbo_;
	unsigned int ibo_;
	size_t number_of_faces_;
	shared_ptr<MeshRenderer> shared_from_;

	void initVAO();
	void uploadData(const shared_ptr<Mesh>& mesh);
	void bindAttributes();

	unsigned int uv_vbo_ = 0;
public:
	MeshRenderer(shared_ptr<Mesh> mesh);
	MeshRenderer(shared_ptr<MeshRenderer> other);
	~MeshRenderer();
	
	void render(unsigned int program, Camera& camera, glm::mat4 globalXform, uint32_t flags);
};

}

#endif // GPU_ENABLED

#endif
