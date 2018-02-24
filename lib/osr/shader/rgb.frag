R"zzz(
#version 430 core
in vec3 fragColor;
in float linearZ;
in vec4 vert_normal;
in vec4 world_position;
layout(location=0) out float outDepth;
layout(location=1) out vec4 outColor;
const float far = 20.0;
const float near = 1.0;
layout(location=16) uniform bool phong;
const vec3 ambient = vec3(0.4, 0.4, 0.4);
const vec3 diffuse = vec3(0.6, 0.6, 0.6);
const vec4 light_position = vec4(0.0, 5.0, 0.0, 1.0);
void main() {
    if (phong) {
	vec4 light_dir = light_position - world_position;
	vec4 normal = normalize(vert_normal);
	float c = clamp(dot(light_dir, normal), 0.0, 1.0);
	outColor = vec4((c * diffuse + ambient) * fragColor, 1.0);
	// outColor = vec4(c, 0.0, 0.0, 1.0);
	// outColor = vec4(abs(normal.xyz), 1.0);
    } else {
        outColor = vec4(fragColor, 1.0);
    }
    outDepth = linearZ;
}
)zzz";
