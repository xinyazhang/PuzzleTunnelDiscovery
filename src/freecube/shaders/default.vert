R"zzz(
#version 330 core
uniform vec4 light_position;
uniform vec3 camera_position;
in vec3 vertex_position;
in vec3 normal;
out vec4 vs_light_direction;
out vec3 vs_normal;
out vec4 vs_camera_direction;
void main() {
	gl_Position = vec4(vertex_position, 1.0);
	vs_light_direction = light_position - gl_Position;
	vs_camera_direction = vec4(camera_position, 1.0) - gl_Position;
	vs_normal = normal;
}
)zzz"
