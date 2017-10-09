#ifndef SCENE_H
#define SCENE_H

#include <string>
#include <vector>

#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <assimp/Importer.hpp>

#define GLM_FORCE_RADIANS
#include <glm/ext.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "camera.h"
#include "mesh.h"
#include "node.h"

namespace osr {

/*
 * Scene
 *
 *      A shareable object to describe a scene among OpenGL contexts.
 *
 *      Note: it is overcomplicated to share everything, so only xform is
 *      shared by making it a reference to master's xform_data object. The
 *      almost read-only member bbox_ and center_ are copied instead of shared
 *      in current practice.
 *
 *      At the same time, meshes_ need to be copied to get different VAO for a
 *      different context.
 */
class Scene {
	size_t vertex_total_number_;
	BoundingBox bbox_;
	glm::vec3 center_;
	glm::mat4 xform_data_;
	glm::mat4& xform_;
	std::shared_ptr<Node> root_;
	std::vector<std::shared_ptr<Mesh>> meshes_;
	std::shared_ptr<Scene> shared_from_;
public:
	Scene();
	Scene(std::shared_ptr<Scene> other);
	virtual ~Scene();

	void load(std::string filename, const glm::vec3* model_color = nullptr);
	void render(GLuint program, Camera& camera, glm::mat4 globalXform);
	void clear();

	/*
	 * Calibration transform matrix shall centralize the scene and scale
	 * it to unit size.
	 */
	glm::mat4 getCalibrationTransform() const { return xform_; }
	BoundingBox getBoundingBox() const { return bbox_; }

	void moveToCenter() { translate(-center_); }
	void resetTransform() { xform_= glm::mat4(); }
	void translate(glm::vec3 offset) {
		xform_= glm::translate(xform_, offset);
	}
	void translate(float x, float y, float z) {
		xform_= glm::translate(xform_, glm::vec3(x, y, z));
	}
	void scale(glm::vec3 factor) { xform_= glm::scale(xform_, factor); }
	void scale(float x, float y, float z) {
		xform_= glm::scale(xform_, glm::vec3(x, y, z));
	}
	void rotate(float rad, glm::vec3 axis) {
		xform_= glm::rotate(xform_, rad, axis);
	}
	void rotate(float rad, float x, float y, float z) {
		xform_= glm::rotate(xform_, rad, glm::vec3(x, y, z));
	}

	void addToCDModel(CDModel& ) const;
private:
	void updateBoundingBox(Node* node, glm::mat4 m);
	void render(GLuint program, Camera& camera, glm::mat4 m, Node* node);
};
}

#endif /* end of include guard: SCENE_H */
