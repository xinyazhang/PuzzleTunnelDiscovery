R"zzz(
#version 330 core
in vec3 fragColor;
in float linearZ;
// layout(location=0) out vec4 outColor;
layout(location=0) out float outDepth;
out vec4 outColor;
const float far = 20.0;
const float near = 1.0;
void main() {
    // gl_FragDepth = (1.0 / gl_FragCoord.w - near) / (far - near);
    // outColor = vec4(fragColor, 1.0);
    outDepth = linearZ;
}
)zzz"
