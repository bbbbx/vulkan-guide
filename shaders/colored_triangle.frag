#version 450

layout(location = 0) in vec3 vColor;

layout(location = 0) out vec4 outFragColor;

void main() {
    outFragColor = vec4(1.0, 1.0, 0.0, 1.0);
    outFragColor = vec4(vColor, 1.0);
}