R"zzz(
#version 430 core
uniform mat4 model;
uniform mat4 view;
uniform mat4 proj;
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec3 inVertNormal;
layout(location = 3) in vec2 inUV;
layout(location = 19) uniform bool is_render_uv_mapping;
out vec3 fragColor;
out gl_PerVertex {
    vec4 gl_Position;
};
out vec4 vert_normal;
out vec4 world_position;
out vec2 uv;
out float linearZ;
void main() {
	if (is_render_uv_mapping) {
		world_position = vec4(inUV, 0.0, 1.0);
		// gl_Position = vec4(inUV, 0.5, 1.0);
		// world_position = model * vec4(inPosition, 1.0);
		// vec4 vm = view * world_position;
		// gl_Position = proj * view * vec4(inUV.x, 0.5, inUV.y, 1.0);
		// gl_Position = proj * view * model * vec4(inUV.x, 0.5, inUV.y, 1.0);
		// world_position = model * vec4(inPosition, 1.0);
		// vec4 vm = view * world_position;
		// gl_Position = proj * vm;
		// gl_Position = proj * view * model * vec4(inPosition, 1.0);
		// gl_Position = proj * view * vec4(inUV.x, inUV.y, 0.0, 1.0);
		gl_Position = vec4(2.0 * inUV - 1.0, 0.0, 1.0);

		vert_normal = vec4(0.0, 0.0, -1.0, 1.0);
		linearZ = 1.0;
		fragColor = vec3(1.0, 1.0, 1.0);
	} else {
		world_position = model * vec4(inPosition, 1.0);
		vec4 vm = view * world_position;
		gl_Position = proj * vm;
		vert_normal = model * vec4(inVertNormal, 0.0);
		// vert_normal = vec4(inVertNormal, 0.0);
		// vert_normal = vec4(0.0, 1.0, 1.0, 0.0);
		linearZ = length(vec3(vm));
		fragColor = inColor;
	}
	uv = inUV;
}
)zzz"
