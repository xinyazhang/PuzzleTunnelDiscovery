R"zzz(
#version 430 core
in vec3 fragBary;
layout(location=0) out float outValue;
// layout(location=0) out vec3 outValue;
void main() {
	if ((min(fragBary.x, fragBary.y), fragBary.z) < 0.0)
		discard;
	if ((max(fragBary.x, fragBary.y), fragBary.z) > 1.0)
		discard;
	outValue = 1.0;
	// outValue = vec3(1.0, 1.0, 1.0);
}
)zzz";
