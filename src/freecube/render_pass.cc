#include <GL/glew.h>
#include "render_pass.h"
#include <iostream>
#include "debuggl.h"
#include <map>

/*
 * For students:
 * 
 * Although RenderPass simplifies the implementation of the reference code.
 * THE USE OF RENDERPASS CLASS IS TOTALLY OPTIONAL.
 * You can implement your system without even take a look of this.
 */

RenderInputMeta::RenderInputMeta()
{
}

RenderInputMeta::RenderInputMeta(int _position,
	            const std::string& _name,
	            const void *_data,
	            size_t _nelements,
	            size_t _element_length,
	            int _element_type)
	:position(_position), name(_name), data(_data),
	nelements(_nelements), element_length(_element_length),
	element_type(_element_type)
{
}

RenderDataInput::RenderDataInput()
{
}

RenderPass::RenderPass(int vao, // -1: create new VAO, otherwise use given VAO
	   const RenderDataInput& input,
	   const std::vector<const char*> shaders, // Order: VS, GS, FS 
	   const std::vector<ShaderUniform> uniforms,
	   const std::vector<const char*> output // Order: 0, 1, 2...
	  )
	: vao_(vao), input_(input), uniforms_(uniforms)
{
	if (vao_ < 0) {
		CHECK_GL_ERROR(glGenVertexArrays(1, (GLuint*)&vao_));
	}
	CHECK_GL_ERROR(glBindVertexArray(vao_));

	// Program first
	vs_ = compileShader(shaders[0], GL_VERTEX_SHADER);
	gs_ = compileShader(shaders[1], GL_GEOMETRY_SHADER);
	fs_ = compileShader(shaders[2], GL_FRAGMENT_SHADER);
	CHECK_GL_ERROR(sp_ = glCreateProgram());
	glAttachShader(sp_, vs_);
	glAttachShader(sp_, fs_);
	if (shaders[1])
		glAttachShader(sp_, gs_);

	// ... and then buffers
	size_t nbuffer = input.getNBuffers();
	if (input.hasIndex())
		nbuffer++;
	glbuffers_.resize(nbuffer);
	CHECK_GL_ERROR(glGenBuffers(nbuffer, glbuffers_.data()));
	for (int i = 0; i < input.getNBuffers(); i++) {
		auto meta = input.getBufferMeta(i);
		CHECK_GL_ERROR(glBindBuffer(GL_ARRAY_BUFFER, glbuffers_[i]));
		CHECK_GL_ERROR(glBufferData(GL_ARRAY_BUFFER,
				meta.getElementSize() * meta.nelements,
				meta.data,
				GL_STATIC_DRAW));
		CHECK_GL_ERROR(glVertexAttribPointer(meta.position,
					meta.element_length,
					meta.element_type,
					GL_FALSE, 0, 0));
		CHECK_GL_ERROR(glEnableVertexAttribArray(meta.position));
		// ... because we need program to bind location
		CHECK_GL_ERROR(glBindAttribLocation(sp_, meta.position, meta.name.c_str()));
	}
	// .. bind output position
	for (size_t i = 0; i < output.size(); i++) {
		CHECK_GL_ERROR(glBindFragDataLocation(sp_, i, output[i]));
	}
	// ... then we can link
	glLinkProgram(sp_);
	CHECK_GL_PROGRAM_ERROR(sp_);

	if (input.hasIndex()) {
		auto meta = input.getIndexMeta();
		CHECK_GL_ERROR(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,
					glbuffers_.back()
					));
		CHECK_GL_ERROR(glBufferData(GL_ELEMENT_ARRAY_BUFFER,
					meta.getElementSize() * meta.nelements,
					meta.data, GL_STATIC_DRAW));
	}
	// after linking uniform locations can be determined
	unilocs_.resize(uniforms.size());
	for (size_t i = 0; i < uniforms.size(); i++) {
		CHECK_GL_ERROR(unilocs_[i] = glGetUniformLocation(sp_, uniforms[i].name.c_str()));
	}
	if (input_.hasMaterial()) {
		createMaterialTexture();
		initMaterialUniform();
	}
}

void RenderPass::initMaterialUniform()
{
	auto float_binder = [](int loc, const void* data) {
		glUniform1fv(loc, 1, (const GLfloat*)data);
	};
	auto vector_binder = [](int loc, const void* data) {
		glUniform4fv(loc, 1, (const GLfloat*)data);
	};
	auto sampler0_binder = [](int loc, const void* data) {
		CHECK_GL_ERROR(glBindSampler(0, (GLuint)(long)data));
	};
	auto texture0_binder = [](int loc, const void* data) {
		CHECK_GL_ERROR(glUniform1i(loc, 0));
		CHECK_GL_ERROR(glActiveTexture(GL_TEXTURE0 + 0));
		CHECK_GL_ERROR(glBindTexture(GL_TEXTURE_2D, (long)data));
		//std::cerr << " bind texture " << long(data) << std::endl;
	};
	material_uniforms_.clear();
	for (size_t i = 0; i < input_.getNMaterials(); i++) {
		auto& ma = input_.getMaterial(i);
		auto diffuse_data = [&ma]() -> const void* {
			return &ma.diffuse[0];
		};
		auto ambient_data = [&ma]() -> const void* {
			return &ma.ambient[0];
		};
		auto specular_data = [&ma]() -> const void* {
			return &ma.specular[0];
		};
		auto shininess_data = [&ma]() -> const void* {
			return &ma.shininess;
		};
		int texid = matexids_[i];
		auto texture_data = [texid]() -> const void* {
			return (const void*)(intptr_t)texid;
		};
		int sam = sampler2d_;
		auto sampler_data = [sam]() -> const void* {
			return (const void*)(intptr_t)sam;
		};
		ShaderUniform diffuse = { "diffuse", vector_binder, diffuse_data };
		ShaderUniform ambient = { "ambient", vector_binder, ambient_data };
		ShaderUniform specular = { "specular", vector_binder, specular_data };
		ShaderUniform shininess = { "shininess", float_binder , shininess_data };
		ShaderUniform texture = { "GL_TEXTURE_2D", texture0_binder , texture_data };
		ShaderUniform sampler = { "textureSampler", sampler0_binder , sampler_data };
		std::vector<ShaderUniform> munis = {diffuse, ambient, specular,
				shininess, texture, sampler};
		material_uniforms_.emplace_back(munis);
	}
	malocs_.clear();
	CHECK_GL_ERROR(malocs_.emplace_back(glGetUniformLocation(sp_, "diffuse")));
	CHECK_GL_ERROR(malocs_.emplace_back(glGetUniformLocation(sp_, "ambient")));
	CHECK_GL_ERROR(malocs_.emplace_back(glGetUniformLocation(sp_, "specular")));
	CHECK_GL_ERROR(malocs_.emplace_back(glGetUniformLocation(sp_, "shininess")));
	CHECK_GL_ERROR(malocs_.emplace_back(glGetUniformLocation(sp_, "textureSampler")));
	CHECK_GL_ERROR(malocs_.emplace_back(glGetUniformLocation(sp_, "textureSampler")));
	std::cerr << "textureSampler location: " << malocs_.back() << std::endl;
}

/*
 * Create textures to gltextures_
 * and assign material specified textures to matexids_
 * 
 * Different materials may share textures
 */
void RenderPass::createMaterialTexture()
{
	CHECK_GL_ERROR(glActiveTexture(GL_TEXTURE0 + 0));
	matexids_.clear();
	std::map<Image*, unsigned> tex2id;
	for (size_t i = 0; i < input_.getNMaterials(); i++) {
		auto& ma = input_.getMaterial(i);
#if 0
		std::cerr << __func__ << " Material " << i << " has texture pointer " << ma.texture.get() << std::endl;
#endif
		if (!ma.texture) {
			matexids_.emplace_back(0);
			continue;
		}
		// Do not create multiple texture for the same data.
		auto iter = tex2id.find(ma.texture.get());
		if (iter != tex2id.end()) {
			matexids_.emplace_back(iter->second);
			continue;
		}

		// Now create and upload texture data
		int w = ma.texture->width;
		int h = ma.texture->height;
		// TODO: enable stride
		// Translate RGB to RGBA for alignment
		std::vector<unsigned int> dummy(w*h);
		const unsigned char* bytes = ma.texture->bytes.data();
		for (int row = 0; row < h; row++) {
			for (int col = 0; col < w; col++) {
				unsigned r = bytes[row*w*3 + col*3];
				unsigned g = bytes[row*w*3 + col*3 + 1];
				unsigned b = bytes[row*w*3 + col*3 + 1];
				dummy[row*w+col] = r | (g << 8) | (b << 16) | (0xFF << 24);
			}
		}
		GLuint tex = 0;
		CHECK_GL_ERROR(glGenTextures(1, &tex));
		CHECK_GL_ERROR(glBindTexture(GL_TEXTURE_2D, tex));
		CHECK_GL_ERROR(glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8,
					w,
					h));
		CHECK_GL_ERROR(glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h,
					GL_RGBA, GL_UNSIGNED_BYTE,
					dummy.data()));
		//CHECK_GL_ERROR(glPixelStorei(GL_UNPACK_ROW_LENGTH, 0));
		std::cerr << __func__ << " load data into texture " << tex <<
			" dim: " << w << " x " << h << std::endl;
		CHECK_GL_ERROR(glBindTexture(GL_TEXTURE_2D, 0));
		matexids_.emplace_back(tex);
		tex2id[ma.texture.get()] = tex;
	}
	CHECK_GL_ERROR(glGenSamplers(1, &sampler2d_));
	CHECK_GL_ERROR(glSamplerParameteri(sampler2d_, GL_TEXTURE_WRAP_S, GL_REPEAT));
	CHECK_GL_ERROR(glSamplerParameteri(sampler2d_, GL_TEXTURE_WRAP_T, GL_REPEAT));
	CHECK_GL_ERROR(glSamplerParameteri(sampler2d_, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
	CHECK_GL_ERROR(glSamplerParameteri(sampler2d_, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
}

RenderPass::~RenderPass()
{
	// TODO: Free resources
}

void RenderPass::updateVBO(int position, const void* data, size_t size)
{
	int bufferid = -1;
	for (int i = 0; i < input_.getNBuffers(); i++) {
		auto meta = input_.getBufferMeta(i);
		if (meta.position == position) {
			bufferid = i;
			break;
		}
	}
	if (bufferid < 0)
		throw __func__+std::string(": error, can't find buffer with position ")+std::to_string(position);
	auto meta = input_.getBufferMeta(bufferid);
	CHECK_GL_ERROR(glBindBuffer(GL_ARRAY_BUFFER, glbuffers_[bufferid]));
	CHECK_GL_ERROR(glBufferData(GL_ARRAY_BUFFER,
				size * meta.getElementSize(),
				data, GL_STATIC_DRAW));
}

void RenderPass::setup()
{
	// Switch to our object VAO.
	CHECK_GL_ERROR(glBindVertexArray(vao_));
	// Use our program.
	CHECK_GL_ERROR(glUseProgram(sp_));

	bind_uniforms(uniforms_, unilocs_);
}

bool RenderPass::renderWithMaterial(int mid)
{
	if (mid >= material_uniforms_.size() || mid < 0)
		return false;
	const auto& mat = input_.getMaterial(mid);
#if 0
	if (!mat.texture)
		return true;
#endif
	auto& matuni = material_uniforms_[mid];
	bind_uniforms(matuni, malocs_);
	CHECK_GL_ERROR(glDrawElements(GL_TRIANGLES, mat.nfaces * 3,
				GL_UNSIGNED_INT,
				(const void*)(mat.offset * 3 * 4)) // Offset is in bytes
	              );
	return true;
}

void RenderPass::bind_uniforms(std::vector<ShaderUniform>& uniforms,
		const std::vector<unsigned>& unilocs)
{
	for (size_t i = 0; i < uniforms.size(); i++) {
		const auto& uni = uniforms[i];
		//std::cerr << "binding " << uni.name << std::endl;
		CHECK_GL_ERROR(uni.binder(unilocs[i], uni.data_source()));
	}
}

unsigned RenderPass::compileShader(const char* source_ptr, int type)
{
	if (!source_ptr)
		return 0;
	auto iter = shader_cache_.find(source_ptr);
	if (iter != shader_cache_.end()) {
		return iter->second;
	}
	GLuint ret = 0;
	CHECK_GL_ERROR(ret = glCreateShader(type));
#if 0
	std::cerr << __func__ << " shader id " << ret << " type " << type << "\tsource:\n" << source_ptr << std::endl;
#endif
	CHECK_GL_ERROR(glShaderSource(ret, 1, &source_ptr, nullptr));
	glCompileShader(ret);
	CHECK_GL_SHADER_ERROR(ret);
	shader_cache_[source_ptr] = ret;
	return ret;
}

void RenderDataInput::assign(int position,
                             const std::string& name,
                             const void *data,
                             size_t nelements,
                             size_t element_length,
                             int element_type)
{
	meta_.emplace_back(position, name, data, nelements, element_length, element_type);
}

void RenderDataInput::assign_index(const void *data, size_t nelements, size_t element_length)
{
	has_index_ = true;
	index_meta_ = {-1, "", data, nelements, element_length, GL_UNSIGNED_INT};
}

void RenderDataInput::useMaterials(const std::vector<Material>& ms)
{
	materials_ = ms;
	for (const auto& ma : ms) {
		std::cerr << "Use Material from " << ma.offset << " size: " << ma.nfaces << std::endl;
	}
}

size_t RenderInputMeta::getElementSize() const
{
	size_t element_size = 4;
	if (element_type == GL_FLOAT)
		element_size = 4;
	else if (element_type == GL_UNSIGNED_INT)
		element_size = 4;
	return element_size * element_length;
}

std::map<const char*, unsigned> RenderPass::shader_cache_;
