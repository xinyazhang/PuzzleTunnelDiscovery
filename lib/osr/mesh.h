/**
 * Copyright (C) 2020 The University of Texas at Austin
 * SPDX-License-Identifier: BSD-3-Clause or GPL-2.0-or-later
 */
#ifndef OSR_MESH_H
#define OSR_MESH_H

#include <glm/ext.hpp>
#include <string>
#include <vector>
#include <memory>
#include <functional>

#include <assimp/scene.h>
#include <assimp/Importer.hpp>
#include <Eigen/Core>

#include "bbox.h"
#include "geometry.h"
#include "quickgl.h"

namespace osr {
class Scene;
class CDModel;

class Mesh {
	friend class Scene;
	std::vector<Vertex> vertices_;
	std::vector<uint32_t> indices_;
	std::shared_ptr<Mesh> shared_from_;
	bool empty_mesh_ = false;
	Eigen::Matrix<float, -1, 2, Eigen::RowMajor> uv_;
public:
	Mesh(std::shared_ptr<Mesh> other);
	Mesh(aiMesh* mesh, glm::vec3 color);
	virtual ~Mesh();

	std::vector<Vertex>& getVertices();
	std::vector<uint32_t>& getIndices();
	const std::vector<Vertex>& getVertices() const;
	const std::vector<uint32_t>& getIndices() const;

	size_t getNumberOfFaces() const;
	void addToCDModel(const glm::mat4&, CDModel&) const;
	bool isEmpty() const { return empty_mesh_; }
	bool hasUV() const { return uv_.rows() > 0; }
	Eigen::Matrix<float, -1, 2, Eigen::RowMajor>& getUV() { return uv_; }
	const Eigen::Matrix<float, -1, 2, Eigen::RowMajor>& getUV() const { return uv_; }
private:
	void init();
};
}

#endif /* end of include guard: MESH_H */
