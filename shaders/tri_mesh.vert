#version 450

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec3 color;

layout(location = 0) out vec3 vColor;

layout (push_constant) uniform constants
{
    vec4 data;
    mat4 render_matrix;
} PushConstants;


void main() {
    gl_Position = PushConstants.render_matrix * vec4(position, 1.0f);
    vColor = color;
}
