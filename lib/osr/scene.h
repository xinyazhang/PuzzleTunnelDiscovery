/**
 * SPDX-FileCopyrightText: Copyright © 2020 The University of Texas at Austin
 * SPDX-FileContributor: Xinya Zhang <xinyazhang@utexas.edu>
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#ifndef OSR_SCENE_H
#define OSR_SCENE_H

#include <string>
#include <vector>
#include <functional>

#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <assimp/Importer.hpp>

#include <glm/ext.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "mesh.h"
#include "node.h"

namespace osr {

class Camera;
class SceneRenderer;
class CDModel;
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
	friend class SceneRenderer;

	size_t vertex_total_number_;
	size_t face_total_number_;
	BoundingBox bbox_;
	glm::vec3 center_;
	glm::vec3 mean_of_vertices_;
	glm::mat4 xform_data_;
	glm::mat4& xform_;
	std::shared_ptr<Node> root_;
	std::vector<std::shared_ptr<Mesh>> meshes_;
	std::shared_ptr<Scene> shared_from_;
	bool has_vertex_normal_;
public:
	Scene();
	Scene(std::shared_ptr<Scene> other);
	virtual ~Scene();

	void load(std::string filename, const glm::vec3* model_color = nullptr);
	void clear();
	void overrideCenter(glm::vec3 c) { center_ = c; }
	glm::vec3 getCenter() const { return center_; }
	glm::vec3 getOMPLCenter() const { return mean_of_vertices_; }

	/*
	 * Calibration transform matrix shall centralize the scene and scale
	 * it to unit size.
	 */
	glm::mat4 getCalibrationTransform() const { return xform_; }
	BoundingBox getBoundingBox() const { return bbox_; }

	void moveToCenter();
	void resetTransform() { xform_= glm::mat4(1.0); }
	void translate(glm::vec3 offset) {
		xform_= glm::translate(xform_, offset);
	}
	void translate(float x, float y, float z) {
		xform_= glm::translate(xform_, glm::vec3(x, y, z));
	}
	void scale(glm::vec3 factor);
	void scale(float x, float y, float z) {
		scale(glm::vec3(x,y,z));
	}
	void rotate(float rad, glm::vec3 axis) {
		xform_= glm::rotate(xform_, rad, axis);
	}
	void rotate(float rad, float x, float y, float z) {
		xform_= glm::rotate(xform_, rad, glm::vec3(x, y, z));
	}

	// Add the transformed geometry to CDModel
	void addToCDModel(CDModel& ) const;

	size_t getNumberOfMeshes() const { return meshes_.size(); }
	void visitMesh(std::function<void(std::shared_ptr<const Mesh>)> ) const;
	bool hasUV() const;
	bool hasVertexNormal() const { return has_vertex_normal_; }

	std::shared_ptr<const Mesh> getUniqueMesh() const;
private:
	void updateBoundingBox(Node* node, glm::mat4 m);
};
}

#endif /* end of include guard: SCENE_H */
