#ifndef RENDER_PASS_H
#define RENDER_PASS_H

/*
 * For students:
 * 
 * This file defines RenderPass class and its associated classes.
 * RenderPass is used to simplify multi-pass rendering code in the
 * reference solution.
 * 
 * However, understanding this class is COMPLETELY OPTIONAL.
 * It's totally OK to ignore this class and cram a bunch of OpenGL
 * function calls in your solution.
 */

#include <vector>
#include <string>
#include <map>
#include <functional>
#include "material.h"

/*
 * ShaderUniform: description of a uniform in a shader program.
 *      name: name
 *      binder: function to bind the uniform
 *      data_source: function to get the data for the uniform
 */
struct ShaderUniform {
	std::string name;
	/*
	 * binder
	 *      argument 0: the location of the uniform
	 *      argument 1: the data pointer returned by data_source function.
	 */
	std::function<void(int, const void*)> binder;
	/*
	 * data_source:
	 *      return: the pointer to the uniform data
	 *      
	 * Hint: DON'T DO THIS
	 *       data_source = []() -> void* { float value = 1.0f; return &f; };
	 *       the value variable will become invalid after returning from
	 *       the lambda function
	 */
	std::function<const void*()> data_source;
};

/*
 * RenderInputMeta: describe one buffer used in some RenderPass
 */
struct RenderInputMeta {
	int position = -1;
	std::string name;
	const void *data = nullptr;
	size_t nelements = 0;
	size_t element_length = 0;
	int element_type = 0;

	size_t getElementSize() const; // simple check: return 12 (3 * 4 bytes) for float3 
	RenderInputMeta();
	RenderInputMeta(int _position,
	            const std::string& _name,
	            const void *_data,
	            size_t _nelements,
	            size_t _element_length,
	            int _element_type);
};

/*
 * RenderDataInput: describe the complete set of buffers used in a RenderPass
 */
class RenderDataInput {
public:
	RenderDataInput();

	/*
	 * assign: assign buffer
	 *      position: glVertexAttribPointer position 
	 *      name: glBindAttribLocation name
	 *      nelements: number of elements
	 *      element_length: element dimension, e.g. for vec3 it's 3
	 *      element_type: GL_FLOAT or GL_UNSIGNED_INT
	 */
	void assign(int position,
	            const std::string& name,
	            const void *data,
	            size_t nelements,
	            size_t element_length,
	            int element_type);
	/*
	 * assign_index: assign the index buffer for vertices
	 * This will bind the data to GL_ELEMENT_ARRAY_BUFFER
	 * The element must be uvec3.
	 */
	void assign_index(const void *data, size_t nelements, size_t element_length);
	/*
	 * useMaterials: assign materials to the input data
	 */
	void useMaterials(const std::vector<Material>& );

	int getNBuffers() const { return int(meta_.size()); }
	RenderInputMeta getBufferMeta(int i) const { return meta_[i]; }
	bool hasIndex() const { return has_index_; }
	RenderInputMeta getIndexMeta() const { return index_meta_; }

	bool hasMaterial() const { return !materials_.empty(); }
	size_t getNMaterials() const { return materials_.size(); }
	const Material& getMaterial(size_t id) const { return materials_[id]; }
	Material& getMaterial(size_t id) { return materials_[id]; }
private:
	std::vector<RenderInputMeta> meta_;
	std::vector<Material> materials_;
	RenderInputMeta index_meta_;
	bool has_index_ = false;
};

class RenderPass {
public:
	/*
	 * Constructor
	 *      vao: the Vertex Array Object, pass -1 to create new
	 *      input: RenderDataInput object
	 *      shaders: array of shaders, leave the second as nullptr if no GS present
	 *      uniforms: array of ShaderUniform objects
	 *      output: the FS output variable name.
	 * RenderPass does not support render-to-texture or multi-target
	 * rendering for now (and you also don't need it).
	 */
	RenderPass(int vao, // -1: create new VAO, otherwise use given VAO
	           const RenderDataInput& input,
	           const std::vector<const char*> shaders, // Order: VS, GS, FS 
	           const std::vector<ShaderUniform> uniforms,
	           const std::vector<const char*> output // Order: 0, 1, 2...
		  );
	~RenderPass();

	unsigned getVAO() const { return unsigned(vao_); }
	void updateVBO(int position, const void* data, size_t nelement);
	void updateIndex(const void* data, size_t nelement);
	void setup();
	/*
 	 * Note: here we don't have an unified render() function, because the
	 * reference solution renders with different primitives
	 * 
	 * However you can freely trianglize everything and add an
	 * render() function
	 */

	/*
	 * renderWithMaterial: render a part of vertex buffer, after binding
	 * corresponding uniforms for Phong shading.
	 */
	bool renderWithMaterial(int i); // return false if material id is invalid

	RenderDataInput& getInput() { return input_; }
private:
	void initMaterialUniform();
	void createMaterialTexture();

	int vao_;
	RenderDataInput input_;
	std::vector<ShaderUniform> uniforms_;
	std::vector<std::vector<ShaderUniform>> material_uniforms_;

	std::vector<unsigned> glbuffers_, unilocs_, malocs_;
	std::vector<unsigned> gltextures_, matexids_;
	unsigned sampler2d_;
	unsigned vs_ = 0, gs_ = 0, fs_ = 0;
	unsigned sp_ = 0;
	
	static unsigned compileShader(const char*, int type);
	static std::map<const char*, unsigned> shader_cache_;

	static void bind_uniforms(std::vector<ShaderUniform>& uniforms, const std::vector<unsigned>& unilocs);
};

#endif
