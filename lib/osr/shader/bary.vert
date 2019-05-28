R"zzz(
#version 430 core
// uniform mat4 model;
// uniform mat4 view;
// uniform mat4 proj;
layout(location = 0) in vec2 inUV;
layout(location = 1) in vec3 inBary;
layout(location = 2) in float inWeight;
out vec2 geoUV;
out vec3 geoBary;
out float geoWeight;
out gl_PerVertex {
    vec4 gl_Position;
};
void main() {
	geoUV = inUV;
	geoBary = inBary;
	geoWeight = inWeight;
	// gl_Position = vec4(2.0 * inUV - 1.0, 0.0, 1.0);
	// gl_Position = vec4(inUV.x, inUV.y, 0.0, 1.0);
	// gl_Position = vec4(inBary, 1.0);
	// vec4 world_position = model * vec4(inBary, 1.0);
	// vec4 vm = view * world_position;
	// gl_Position = proj * vm;
}
)zzz"
