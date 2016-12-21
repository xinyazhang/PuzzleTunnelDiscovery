R"zzz(
#version 330 core
uniform vec4 diffuse;
uniform float shininess;
uniform float alpha;
uniform sampler2D textureSampler;
out vec4 fragment_color;

void main() {
	vec3 color = vec3(diffuse);
	fragment_color = vec4(color, alpha);
}
)zzz"
