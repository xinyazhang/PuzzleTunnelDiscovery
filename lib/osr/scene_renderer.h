#ifndef OFF_SCREEN_RENDERING_SCENE_RENDERER_H
#define OFF_SCREEN_RENDERING_SCENE_RENDERER_H

#if GPU_ENABLED

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <GL/glew.h>
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

	void render(GLuint program, Camera& camera, glm::mat4 globalXform);
private:
	void render(GLuint program, Camera& camera, glm::mat4 m, Node* node);
};

}

#endif // GPU_ENABLED

#endif
