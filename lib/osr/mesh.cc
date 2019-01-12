#include "mesh.h"
#include "cdmodel.h"
#include <algorithm>
#include <fcl/narrowphase/collision.h>
#include <glm/gtx/io.hpp>

namespace osr {

glm::vec3 to_glm_vec3(aiVector3D vec)
{
	return glm::vec3(vec.x, vec.y, vec.z);
}

Mesh::Mesh(std::shared_ptr<Mesh> other)
	:shared_from_(other)
{
}

Mesh::Mesh(aiMesh* mesh, glm::vec3 color)
	:shared_from_(nullptr)
{
	size_t NV = mesh->mNumVertices;
	if (mesh->HasNormals()) {
		for (size_t i = 0; i < NV; i++) {
			vertices_.emplace_back(
				to_glm_vec3(mesh->mVertices[i]),  // position
				color,
				to_glm_vec3(mesh->mNormals[i])  // normal
				);
			// std::cerr << "Mesh Normal: " << to_glm_vec3(mesh->mNormals[i]) << '\n';
		}
	} else {
		for (size_t i = 0; i < NV; i++) {
			vertices_.emplace_back(to_glm_vec3(mesh->mVertices[i]),
			                       color);
		}
	}
	if (mesh->HasTextureCoords(0)) {
		uv_.resize(NV, 2);
		for (size_t i = 0; i < NV; i++) {
			uv_(i,0) = mesh->mTextureCoords[0][i][0];
			uv_(i,1) = mesh->mTextureCoords[0][i][1]; 
			// std::cerr << "Loading UV " << uv_.row(i) << std::endl;
		}
		std::cerr << "Load UV: " << uv_.rows() << std::endl;
	} else {
		uv_.resize(0, Eigen::NoChange);
		std::cerr << "UV Not Found" << std::endl;
	}
	for (size_t i = 0; i < mesh->mNumFaces; i++) {
		aiFace& face = mesh->mFaces[i];
		if (face.mNumIndices == 3) {
			for (size_t j = 0; j < face.mNumIndices; j++)
				indices_.emplace_back(face.mIndices[j]);
		}
	}
	empty_mesh_ = (indices_.size() == 0);
}

Mesh::~Mesh()
{
}

std::vector<Vertex>& Mesh::getVertices()
{
	if (shared_from_)
		return shared_from_->vertices_;
	return vertices_;
}

std::vector<uint32_t>& Mesh::getIndices()
{
	if (shared_from_)
		return shared_from_->indices_;
	return indices_;
}

const std::vector<Vertex>& Mesh::getVertices() const
{
	return const_cast<Mesh*>(this)->getVertices();
}

const std::vector<uint32_t>& Mesh::getIndices() const
{
	return const_cast<Mesh*>(this)->getIndices();
}

size_t Mesh::getNumberOfFaces() const
{
	if (shared_from_)
		return shared_from_->indices_.size();
	return indices_.size();
}

void Mesh::addToCDModel(const glm::mat4& m, CDModel& model) const
{
	model.addVF(m, vertices_, indices_);
}


}
