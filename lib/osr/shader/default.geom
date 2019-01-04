R"zzz(
#version 430 core
layout (triangles) in;
layout (triangle_strip, max_vertices = 3) out;
layout(location = 20) uniform bool flat_surface;
in vec3 vso_color[];
in vec4 vso_vert_normal[];
in vec4 vso_world_position[];
in vec2 vso_uv[];
in float vso_linearZ[];

out vec3 fragColor;
out vec4 vert_normal;
out vec4 world_position;
out vec2 uv;
out float linearZ;
void main() {
	vec4 flat_normal;
	if (flat_surface) {
		vec3 a = vso_world_position[0].xyz;
		vec3 b = vso_world_position[1].xyz;
		vec3 c = vso_world_position[2].xyz;
		vec3 u = normalize(b - a);
		vec3 v = normalize(c - a);
		flat_normal = vec4(normalize(cross(u, v)), 0.0);
	}
	int n;
	for (n = 0; n < gl_in.length(); n++) {
		// Passing thru variables
		fragColor = vso_color[n];
		world_position = vso_world_position[n];
		uv = vso_uv[n];
		linearZ = vso_linearZ[n];
		gl_Position = gl_in[n].gl_Position;
		if (flat_surface) {
			vert_normal = flat_normal;
		} else {
			vert_normal = vso_vert_normal[n];
		}
                EmitVertex();
	}
	EndPrimitive();
}
)zzz"

