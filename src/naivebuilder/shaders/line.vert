R"zzz(
#version 330 core
uniform mat4 projection;
uniform mat4 model;
uniform mat4 view;

in vec3 vertex_position;

void main() {
	gl_Position = projection * view * model * vec4(vertex_position, 1.0);
}
)zzz"
