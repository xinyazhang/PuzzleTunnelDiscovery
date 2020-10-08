/**
 * SPDX-FileCopyrightText: Copyright © 2020 The University of Texas at Austin
 * SPDX-FileContributor: Xinya Zhang <xinyazhang@utexas.edu>
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#if GPU_ENABLED

#include "scene_renderer.h"
#include "mesh_renderer.h"
#include "osr_render.h"
#include "node.h"
#include "camera.h"
#include "scene.h"
#include "pngimage.h"
#include <strman/suffix.h>

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
	if (tex_)
		glDeleteTextures(1, &tex_);
}

void SceneRenderer::probe_texture(const std::string& fn)
{
	std::string tex_fn = strman::replace_suffix(fn, ".ply", ".png");
	std::string tex2_fn = strman::replace_suffix(fn, ".obj", ".png");
	if (tex_fn.empty()) {
		if (tex2_fn.empty()) {
			tex_data_.clear();
			return;
		}
		tex_fn = tex2_fn;
	}
	load_texture(tex_fn);
}

void SceneRenderer::load_texture(const std::string& tex_fn)
{
	int pc;
	tex_data_ = readPNG(tex_fn.c_str(), tex_w_, tex_h_, &pc);
	if (tex_data_.empty())
		return;
	auto iformat = GL_RGBA8;
	auto dformat = GL_RGBA;
	if (pc == 3) {
		iformat = GL_RGB8;
		dformat = GL_RGB;
	}
	if (tex_) {
		CHECK_GL_ERROR(glDeleteTextures(1, &tex_));
		tex_ = 0;
	}
	if (!tex_) {
		CHECK_GL_ERROR(glGenTextures(1, &tex_));
	}
	CHECK_GL_ERROR(glBindTexture(GL_TEXTURE_2D, tex_));
	CHECK_GL_ERROR(glTexStorage2D(GL_TEXTURE_2D, 1, iformat, tex_w_, tex_h_));
	CHECK_GL_ERROR(glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, tex_w_, tex_h_,
				       dformat, GL_UNSIGNED_BYTE,
				       tex_data_.data()));
	CHECK_GL_ERROR(glBindTexture(GL_TEXTURE_2D, 0));
	if (!sam_) {
		CHECK_GL_ERROR(glGenSamplers(1, &sam_));
		CHECK_GL_ERROR(glSamplerParameteri(sam_, GL_TEXTURE_WRAP_S, GL_REPEAT));
		CHECK_GL_ERROR(glSamplerParameteri(sam_, GL_TEXTURE_WRAP_T, GL_REPEAT));
		CHECK_GL_ERROR(glSamplerParameteri(sam_, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
		CHECK_GL_ERROR(glSamplerParameteri(sam_, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
	}
}

void
SceneRenderer::render(GLuint program, Camera& camera, glm::mat4 m, uint32_t flags)
{
	if (tex_ && (flags & Renderer::HAS_NTR_RENDERING)) {
		CHECK_GL_ERROR(glActiveTexture(GL_TEXTURE0));
		CHECK_GL_ERROR(glBindTexture(GL_TEXTURE_2D, tex_));
		CHECK_GL_ERROR(glBindSampler(0, sam_));
	} else {
		CHECK_GL_ERROR(glBindTexture(GL_TEXTURE_2D, 0));
	}
	// render(program, camera, m * xform, root);
	for (const auto& mr : renderers_) {
		/*
		 * Do not call empty MeshRenderer
		 */
		if (!mr)
			continue;
		mr->render(program, camera,
		           m * scene_->getCalibrationTransform(),
		           flags);
	}
	if (tex_ && (flags & Renderer::HAS_NTR_RENDERING)) {
		CHECK_GL_ERROR(glBindTexture(GL_TEXTURE_2D, 0));
	}
}

void
SceneRenderer::render(GLuint program, Camera& camera, glm::mat4 m, Node* node, uint32_t flags)
{
	glm::mat4 xform = m * node->xform;
#if 0
    if (node->meshes.size() > 0)
        std::cout << "matrix: " << std::endl << glm::to_string(xform) << std::endl;
#endif
	for (auto i : node->meshes) {
		auto mr = renderers_[i];
		mr->render(program, camera, xform, flags);
	}
	for (auto child : node->nodes) {
		render(program, camera, xform, child.get(), flags);
	}
}

}

#endif // GPU_ENABLED
