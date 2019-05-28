R"zzz(
#version 430 core
layout (triangles) in;
layout (triangle_strip, max_vertices = 3) out;
in vec2 geoUV[];
in vec3 geoBary[];
in float geoWeight[];
out vec3 fragBary;
out float fragWeight;
void main() {
	int n;
	for (n = 0; n < gl_in.length(); n++) {
		vec2 blend = geoUV[0] * geoBary[n].x;
		blend += geoUV[1] * geoBary[n].y;
		blend += geoUV[2] * geoBary[n].z;
		gl_Position = vec4(2.0 * blend - 1.0, 0.0, 1.0);
		// gl_Position = vec4(2.0 * geoUV[n] - 1.0, 0.0, 1.0);
		fragBary = geoBary[n];
		fragWeight = geoWeight[n];
                EmitVertex();
	}
	EndPrimitive();
}
)zzz"
