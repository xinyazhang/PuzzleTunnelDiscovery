#ifndef MATERIAL_H
#define MATERIAL_H

#include "image.h"
#include <glm/glm.hpp>
#include <memory>

/*
 * PMD format groups faces according to their materials.
 * Each material was assigned to a continuous sequence of faces.
 * 
 * A material consists of Phong shading model data and optionally a texture.
 * 
 * Note: you need to bypass Phong shading model to get correct results if the
 * texture presented in a material, which can be done by checking the texture
 * color is zero or not in the shader.
 * 
 * Alternatively in theory, you can also call textureSize in GLSL to check if
 * the texture size is non-zero. However this method doesn't work here...
 */
struct Material {
	// Phong shading model
	glm::vec4 diffuse, ambient, specular;
	float shininess;
	std::shared_ptr<Image> texture; // Texture for current material, can be null.

	size_t offset; // This material applies to faces starting from offset.
	size_t nfaces; // This material applies to nfaces faces.
};

#endif
