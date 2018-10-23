R"zzz(
#version 330 core
uniform mat4 model;
uniform mat4 view;
uniform mat4 proj;
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec3 inVertNormal;
layout(location = 3) in vec2 inUV;
out vec3 fragColor;
out gl_PerVertex {
    vec4 gl_Position;
};
out vec4 vert_normal;
out vec4 world_position;
out vec2 uv;
out float linearZ;
void main() {
    world_position = model * vec4(inPosition, 1.0);
    vec4 vm = view * world_position;
    gl_Position = proj * vm;
    vert_normal = model * vec4(inVertNormal, 0.0);
    // vert_normal = vec4(inVertNormal, 0.0);
    // vert_normal = vec4(0.0, 1.0, 1.0, 0.0);
    linearZ = length(vec3(vm));
    fragColor = inColor;
    uv = inUV;
}
)zzz"
