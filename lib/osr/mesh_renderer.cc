#if GPU_ENABLED

#include "mesh_renderer.h"
#include "osr_render.h"
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

	if (mesh->hasUV()) {
		CHECK_GL_ERROR(glGenBuffers(1, &uv_vbo_));
		CHECK_GL_ERROR(glBindBuffer(GL_ARRAY_BUFFER, uv_vbo_));
		CHECK_GL_ERROR(glBufferData(GL_ARRAY_BUFFER,
		                            sizeof(float) * mesh->getUV().size(),
		                            mesh->getUV().data(), GL_STATIC_DRAW));
		std::cerr << "Upload " << mesh->getUV().size() << " elements to UV VBO\n";
		CHECK_GL_ERROR(glBindBuffer(GL_ARRAY_BUFFER, 0));
	}
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

	offset = (void*)offsetof(Vertex, normal);
	CHECK_GL_ERROR(glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE,
					     sizeof(Vertex),
					     offset));  // vertex color
	CHECK_GL_ERROR(glEnableVertexAttribArray(2));
	CHECK_GL_ERROR(glBindBuffer(GL_ARRAY_BUFFER, 0));

	CHECK_GL_ERROR(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo_));

	if (uv_vbo_) {
		CHECK_GL_ERROR(glBindBuffer(GL_ARRAY_BUFFER, uv_vbo_));
		CHECK_GL_ERROR(glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE,
		                                     sizeof(float) * 2,
		                                     0));  // vertex color
		CHECK_GL_ERROR(glEnableVertexAttribArray(3));
		CHECK_GL_ERROR(glBindBuffer(GL_ARRAY_BUFFER, 0));
		std::cerr << "Bind UV VBO\n";
	} else {
		CHECK_GL_ERROR(glDisableVertexAttribArray(3));
	}

	CHECK_GL_ERROR(glBindVertexArray(0));
}


void
MeshRenderer::render(GLuint program, Camera& camera, glm::mat4 globalXform, uint32_t flags)
{
	camera.uniform(program, globalXform);
	CHECK_GL_ERROR(glBindVertexArray(vao_));

#if 0
	CHECK_GL_ERROR(glBindAttribLocation(program, 0, "inPosition"));
	CHECK_GL_ERROR(glBindAttribLocation(program, 1, "inColor"));
#endif
	if (uv_vbo_) {
		CHECK_GL_ERROR(glEnableVertexAttribArray(3));
	} else {
		CHECK_GL_ERROR(glDisableVertexAttribArray(3));
	}

	CHECK_GL_ERROR(glDrawElements(GL_TRIANGLES, number_of_faces_,
	                              GL_UNSIGNED_INT, 0));

	CHECK_GL_ERROR(glBindBuffer(GL_ARRAY_BUFFER, 0));
}

}

#endif // GPU_ENABLED
