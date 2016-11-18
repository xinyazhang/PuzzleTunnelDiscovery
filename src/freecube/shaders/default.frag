R"zzz(
#version 330 core
in vec3 face_normal;
in vec3 vertex_normal;
in vec4 light_direction;
in vec4 camera_direction;
uniform vec4 diffuse;
// uniform vec4 ambient;
// uniform vec4 specular;
uniform float shininess;
uniform float alpha;
uniform sampler2D textureSampler;
out vec4 fragment_color;

void main() {
	vec3 color = vec3(diffuse);
	float dot_nl = dot(normalize(light_direction), normalize(vec4(vertex_normal, 0.0)));
	dot_nl = clamp(dot_nl + 0.1, 0.0, 1.0);
	// vec4 spec = specular * pow(max(0.0, dot(reflect(-light_direction, vertex_normal), camera_direction)), shininess);
	// color = clamp(dot_nl * color + vec3(ambient) + vec3(spec), 0.0, 1.0);
	color = dot_nl * color;
	fragment_color = vec4(color, alpha);
	//fragment_color = vec4(1.0, 1.0, 1.0, 1.0);
}
)zzz"
