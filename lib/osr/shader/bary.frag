R"zzz(
#version 430 core
in vec3 fragBary;
layout(location=0) out float outValue;
// layout(location=0) out vec3 outValue;
void main() {
	float m = min(min(fragBary.x, fragBary.y), fragBary.z);
	float ma = max(max(fragBary.x, fragBary.y), fragBary.z);
#if 0
	if (m < 0.1)
		outValue = 1.0;
	else
		outValue = 0.5;
#else
	if (m < 0.0) {
		discard;
	} else if (ma > 1.0) {
		discard;
	} else {
		outValue = 1.0;
	}
#endif
	// outValue = vec3(1.0, 1.0, 1.0);
}
)zzz";
