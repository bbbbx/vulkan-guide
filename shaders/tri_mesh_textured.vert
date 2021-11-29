#version 460

layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec3 color;
layout (location = 3) in vec2 uv;

layout (set = 0, binding = 0) uniform CameraBuffer {
    mat4 view;
    mat4 projection;
    mat4 viewproj;
} cameraData;

// layout ( push_constant ) uniform constants {
//     vec4 data;
//     mat4 render_matrix;
// } PushConstants;

struct ObjectData {
    mat4 model;
};
layout (std140, set = 1, binding = 0) readonly buffer ObjectBuffer {
    ObjectData objects[];
} objectBuffer;

layout (location = 0) out vec3 vColor;
layout (location = 1) out vec2 vUv;

void main() {
    mat4 modelMatrix = objectBuffer.objects[gl_BaseInstance].model;
    gl_Position = cameraData.viewproj * modelMatrix * vec4(position, 1.0f);
    vColor = color;
    vUv = uv;
}