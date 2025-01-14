#version 450

layout (set = 0, binding = 1) uniform SceneData {
    vec4 fogColor; // w is for exponent
    vec4 fogDistances; // x for near, y for far, zw unused
    vec4 ambientColor;
    vec4 sunlightDirection; // w for sun power
    vec4 sunlightColor;
} sceneData;

layout(location = 0) in vec3 vColor;

layout(location = 0) out vec4 outFragColor;

void main() {
    outFragColor = vec4(vColor + sceneData.ambientColor.rgb, 1.0);
}