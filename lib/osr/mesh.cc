#include "mesh.h"

namespace osr {
glm::vec3 to_glm_vec3(aiVector3D vec)
{
	return glm::vec3(vec.x, vec.y, vec.z);
}

Mesh::Mesh(std::shared_ptr<Mesh> other)
	:shared_from_(other)
{
	vbo_ = other->vbo_;
	ibo_ = other->ibo_;
	vao_ = 0;
	empty_mesh_ = other->empty_mesh_;
	init();
}

Mesh::Mesh(aiMesh* mesh, glm::vec3 color)
	:shared_from_(nullptr)
{
	for (size_t i = 0; i < mesh->mNumVertices; i++) {
		vertices_.push_back(
		        {to_glm_vec3(mesh->mVertices[i]),  // position
		         color});
	}
	for (size_t i = 0; i < mesh->mNumFaces; i++) {
		aiFace& face = mesh->mFaces[i];
		if (face.mNumIndices == 3) {
			for (size_t j = 0; j < face.mNumIndices; j++)
				indices_.emplace_back(face.mIndices[j]);
		}
	}
	empty_mesh_ = (indices_.size() == 0);
	init();
}

Mesh::~Mesh()
{
	CHECK_GL_ERROR(glBindVertexArray(0));
	CHECK_GL_ERROR(glDeleteVertexArrays(1, &vao_));
	if (!shared_from_) {
		CHECK_GL_ERROR(glDeleteBuffers(1, &vbo_));
		CHECK_GL_ERROR(glDeleteBuffers(1, &ibo_));
	}
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

size_t Mesh::getNumberOfFaces() const
{
	if (shared_from_)
		return shared_from_->indices_.size();
	return indices_.size();
}

void Mesh::render(GLuint program, Camera& camera, glm::mat4 globalXform)
{
	if (empty_mesh_)
		return;
	camera.uniform(program, globalXform);
	CHECK_GL_ERROR(glBindVertexArray(vao_));

	CHECK_GL_ERROR(glBindAttribLocation(program, 0, "inPosition"));
	CHECK_GL_ERROR(glBindAttribLocation(program, 1, "inColor"));

	CHECK_GL_ERROR(glDrawElements(GL_TRIANGLES, getNumberOfFaces(),
	                              GL_UNSIGNED_INT, 0));

	CHECK_GL_ERROR(glBindBuffer(GL_ARRAY_BUFFER, 0));
}


/*
 * Mesh::init()
 *
 *      Create VAO and bind mesh data (VBO/IBO) to it.
 *      We may need to create VBO/IBO if not copying from a master Mesh.
 */
void Mesh::init()
{
	if (empty_mesh_)
		return;
	CHECK_GL_ERROR(glGenVertexArrays(1, &vao_));
	CHECK_GL_ERROR(glBindVertexArray(vao_));
	std::cerr << __func__ << " CREATE VAO " << vao_ << std::endl;
	if (!shared_from_) {
		CHECK_GL_ERROR(glGenBuffers(1, &vbo_));
		CHECK_GL_ERROR(glBindBuffer(GL_ARRAY_BUFFER, vbo_));
		CHECK_GL_ERROR(glBufferData(GL_ARRAY_BUFFER,
					    sizeof(Vertex) * getVertices().size(),
					    getVertices().data(), GL_STATIC_DRAW));
		std::cerr << __func__ << " CREATE VBO: " << vbo_ << std::endl;
	} else {
		std::cerr << __func__ << " REUSE VBO: " << vbo_ << std::endl;
		CHECK_GL_ERROR(glBindBuffer(GL_ARRAY_BUFFER, vbo_));
	}
	void* offset;
	offset = (void*)offsetof(Vertex, position);
	// std::cerr << "offset of position: " << offset << std::endl;
	CHECK_GL_ERROR(glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE,
					     sizeof(Vertex), offset));
	CHECK_GL_ERROR(glEnableVertexAttribArray(0));

	offset = (void*)offsetof(Vertex, color);
	// std::cerr << "offset of color: " << offset << std::endl;
	CHECK_GL_ERROR(glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE,
					     sizeof(Vertex),
					     offset));  // vertex color
	CHECK_GL_ERROR(glEnableVertexAttribArray(1));
	CHECK_GL_ERROR(glBindBuffer(GL_ARRAY_BUFFER, 0));

	if (!shared_from_) {
		CHECK_GL_ERROR(glGenBuffers(1, &ibo_));
		CHECK_GL_ERROR(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo_));
		CHECK_GL_ERROR(glBufferData(GL_ELEMENT_ARRAY_BUFFER,
					    sizeof(uint32_t) * getIndices().size(),
					    getIndices().data(), GL_STATIC_DRAW));
	} else {
		CHECK_GL_ERROR(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo_));
	}

	CHECK_GL_ERROR(glBindVertexArray(0));
}

}
