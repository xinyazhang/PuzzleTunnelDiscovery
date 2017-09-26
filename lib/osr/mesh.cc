#include "mesh.h"

namespace osr {
glm::vec3 to_glm_vec3(aiVector3D vec)
{
	return glm::vec3(vec.x, vec.y, vec.z);
}

Mesh::Mesh(Mesh& rhs)
{
	vertices = rhs.vertices;
	indices  = rhs.indices;
	vbo      = rhs.vbo;
	ibo      = rhs.ibo;
	vao      = rhs.vao;
}

Mesh::Mesh(aiMesh* mesh, glm::vec3 color)
{
	CHECK_GL_ERROR(glGenVertexArrays(1, &vao));
	CHECK_GL_ERROR(glGenBuffers(1, &vbo));
	CHECK_GL_ERROR(glGenBuffers(1, &ibo));

	for (size_t i = 0; i < mesh->mNumVertices; i++) {
		vertices.push_back(
		        {to_glm_vec3(mesh->mVertices[i]),  // position
		         color});
	}
	for (size_t i = 0; i < mesh->mNumFaces; i++) {
		aiFace& face = mesh->mFaces[i];
		if (face.mNumIndices == 3) {
			for (size_t j = 0; j < face.mNumIndices; j++)
				indices.push_back(face.mIndices[j]);
		}
	}
	init();
}

Mesh::~Mesh()
{
	CHECK_GL_ERROR(glBindVertexArray(0));
	CHECK_GL_ERROR(glDeleteBuffers(1, &vbo));
	CHECK_GL_ERROR(glDeleteBuffers(1, &ibo));
	CHECK_GL_ERROR(glDeleteVertexArrays(1, &vao));
}

void Mesh::render(GLuint program, Camera& camera, glm::mat4 globalXform)
{
	if (indices.size() == 0)
		return;
	camera.uniform(program, globalXform);
	CHECK_GL_ERROR(glBindVertexArray(vao));

	CHECK_GL_ERROR(glBindAttribLocation(program, 0, "inPosition"));
	CHECK_GL_ERROR(glBindAttribLocation(program, 1, "inColor"));

	CHECK_GL_ERROR(glDrawElements(GL_TRIANGLES, indices.size(),
	                              GL_UNSIGNED_INT, 0));

	CHECK_GL_ERROR(glBindBuffer(GL_ARRAY_BUFFER, 0));
}

void Mesh::init()
{
	// std::cerr << "Binding VAO: " << vao << std::endl;
	CHECK_GL_ERROR(glBindVertexArray(vao));
	if (indices.size() == 0)
		return;
	void* offset;
	CHECK_GL_ERROR(glGenBuffers(1, &vbo));
	CHECK_GL_ERROR(glBindBuffer(GL_ARRAY_BUFFER, vbo));
	CHECK_GL_ERROR(glBufferData(GL_ARRAY_BUFFER,
	                            sizeof(Vertex) * vertices.size(),
	                            vertices.data(), GL_STATIC_DRAW));

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

	CHECK_GL_ERROR(glGenBuffers(1, &ibo));
	CHECK_GL_ERROR(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo));
	CHECK_GL_ERROR(glBufferData(GL_ELEMENT_ARRAY_BUFFER,
	                            sizeof(uint32_t) * indices.size(),
	                            indices.data(), GL_STATIC_DRAW));

	CHECK_GL_ERROR(glBindVertexArray(0));
}
}
