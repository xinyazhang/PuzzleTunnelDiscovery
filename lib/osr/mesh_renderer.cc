#if GPU_ENABLED

#include "mesh_renderer.h"
#include "camera.h"
#include "mesh.h"

namespace osr {

MeshRenderer::MeshRenderer(shared_ptr<Mesh> mesh)
{
	initVAO();
	uploadData(mesh);
	bindAttributes();

	number_of_faces_ = mesh->getNumberOfFaces();
}

MeshRenderer::MeshRenderer(shared_ptr<MeshRenderer> other)
	:shared_from_(other)
{
	initVAO();
	vbo_ = other->vbo_;
	ibo_ = other->ibo_;
	bindAttributes();

	number_of_faces_ = other->number_of_faces_;
}

MeshRenderer::~MeshRenderer()
{
	CHECK_GL_ERROR(glBindVertexArray(0));
	CHECK_GL_ERROR(glDeleteVertexArrays(1, &vao_));
	if (!shared_from_) {
		CHECK_GL_ERROR(glDeleteBuffers(1, &vbo_));
		CHECK_GL_ERROR(glDeleteBuffers(1, &ibo_));
	}
}

void
MeshRenderer::initVAO()
{
	CHECK_GL_ERROR(glGenVertexArrays(1, &vao_));
}

void
MeshRenderer::uploadData(const shared_ptr<Mesh>& mesh)
{
	CHECK_GL_ERROR(glGenBuffers(1, &vbo_));
	CHECK_GL_ERROR(glBindBuffer(GL_ARRAY_BUFFER, vbo_));
	CHECK_GL_ERROR(glBufferData(GL_ARRAY_BUFFER,
	                            sizeof(Vertex) * mesh->getVertices().size(),
	                            mesh->getVertices().data(), GL_STATIC_DRAW));

	CHECK_GL_ERROR(glBindVertexArray(vao_));
	CHECK_GL_ERROR(glGenBuffers(1, &ibo_));
	CHECK_GL_ERROR(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo_));
	CHECK_GL_ERROR(glBufferData(GL_ELEMENT_ARRAY_BUFFER,
	                            sizeof(uint32_t) * mesh->getIndices().size(),
	                            mesh->getIndices().data(), GL_STATIC_DRAW));
	CHECK_GL_ERROR(glBindVertexArray(0));
}

void
MeshRenderer::bindAttributes()
{
	CHECK_GL_ERROR(glBindVertexArray(vao_));
	CHECK_GL_ERROR(glBindBuffer(GL_ARRAY_BUFFER, vbo_));
	void* offset;
	offset = (void*)offsetof(Vertex, position);
	CHECK_GL_ERROR(glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE,
					     sizeof(Vertex), offset));
	CHECK_GL_ERROR(glEnableVertexAttribArray(0));

	offset = (void*)offsetof(Vertex, color);
	CHECK_GL_ERROR(glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE,
					     sizeof(Vertex),
					     offset));  // vertex color
	CHECK_GL_ERROR(glEnableVertexAttribArray(1));
	CHECK_GL_ERROR(glBindBuffer(GL_ARRAY_BUFFER, 0));

	CHECK_GL_ERROR(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo_));
	CHECK_GL_ERROR(glBindVertexArray(0));
}


void
MeshRenderer::render(GLuint program, Camera& camera, glm::mat4 globalXform)
{
	camera.uniform(program, globalXform);
	CHECK_GL_ERROR(glBindVertexArray(vao_));

	CHECK_GL_ERROR(glBindAttribLocation(program, 0, "inPosition"));
	CHECK_GL_ERROR(glBindAttribLocation(program, 1, "inColor"));

	CHECK_GL_ERROR(glDrawElements(GL_TRIANGLES, number_of_faces_,
	                              GL_UNSIGNED_INT, 0));

	CHECK_GL_ERROR(glBindBuffer(GL_ARRAY_BUFFER, 0));
}

}

#endif // GPU_ENABLED
