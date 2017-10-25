#include "scene_renderer.h"
#include "mesh_renderer.h"
#include "node.h"
#include "camera.h"
#include "scene.h"

namespace osr {

SceneRenderer::SceneRenderer(shared_ptr<Scene> scene)
	:scene_(scene)
{
	/*
	 * Skip empty meshes
	 */
	for (auto mesh : scene->meshes_) {
		if (mesh->isEmpty())
			renderers_.emplace_back(nullptr);
		else
			renderers_.emplace_back(new MeshRenderer(mesh));
	}
}

SceneRenderer::SceneRenderer(shared_ptr<SceneRenderer> other)
	:shared_from_(other), scene_(other->scene_)
{
	/*
	 * Do not copy empty MeshRenderer
	 * 
	 * mr means "Mesh Renderer"
	 */
	for (auto mr : other->renderers_) {
		if (!mr)
			renderers_.emplace_back(nullptr);
		else
			renderers_.emplace_back(new MeshRenderer(mr));
	}
}

SceneRenderer::~SceneRenderer()
{
}

void
SceneRenderer::render(GLuint program, Camera& camera, glm::mat4 m)
{
	// render(program, camera, m * xform, root);
	for (const auto& mr : renderers_) {
		/*
		 * Do not call empty MeshRenderer
		 */
		if (!mr)
			continue;
		mr->render(program, camera,
		           m * scene_->getCalibrationTransform());
	}
}

void
SceneRenderer::render(GLuint program, Camera& camera, glm::mat4 m, Node* node)
{
	glm::mat4 xform = m * node->xform;
#if 0
    if (node->meshes.size() > 0)
        std::cout << "matrix: " << std::endl << glm::to_string(xform) << std::endl;
#endif
	for (auto i : node->meshes) {
		auto mr = renderers_[i];
		mr->render(program, camera, xform);
	}
	for (auto child : node->nodes) {
		render(program, camera, xform, child.get());
	}
}

}
