#ifndef OFF_SCREEN_RENDERING_SCENE_RENDERER_H
#define OFF_SCREEN_RENDERING_SCENE_RENDERER_H

#if GPU_ENABLED

#include <glm/glm.hpp>
#include "quickgl.h"
#include <memory>
#include <vector>

namespace osr {

using std::shared_ptr;
class Scene;
class MeshRenderer;
class Camera;
class Node;

class SceneRenderer {
	shared_ptr<Scene> scene_;
	shared_ptr<SceneRenderer> shared_from_;
	std::vector<shared_ptr<MeshRenderer>> renderers_;
public:
	/*
	 * SceneRenderer will create new GL objects if created from
	 * Scene, otherwise it shares existing GL objects by exploiting the
	 * sharing mechanism in MeshRenderer
	 */
	SceneRenderer(shared_ptr<Scene> scene);
	SceneRenderer(shared_ptr<SceneRenderer> other);
	~SceneRenderer();

	void probe_texture(const std::string& fn);
	void load_texture(const std::string& fn);
	void render(GLuint program, Camera& camera, glm::mat4 globalXform, uint32_t flags);
private:
	void render(GLuint program, Camera& camera, glm::mat4 m, Node* node, uint32_t flags);

	std::vector<uint8_t> tex_data_;
	int tex_w_, tex_h_;
	GLuint tex_ = 0;
	GLuint sam_ = 0;
};

}

#endif // GPU_ENABLED

#endif
