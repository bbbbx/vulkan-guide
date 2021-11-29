// vulkan_guide.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <vector>
#include <functional>
#include <deque>

#include <glm/glm.hpp>

#include "vk_mesh.h"

#include <string>
#include <unordered_map>

constexpr unsigned int FRAME_OVERLAP = 2;

struct GPUCameraData {
    glm::mat4 view;
    glm::mat4 projection;
    glm::mat4 viewproj;
};

struct GPUSceneData {
    glm::vec4 fogColor;  // w is exponent
    glm::vec4 fogDistances; // x for min, y for max, zw unused
    glm::vec4 ambientColor;
    glm::vec4 sunlightDirection; // w for sun power
    glm::vec4 sunlightColor;
};

struct GPUObjectData {
	glm::mat4 modelMatrix;
};

struct FrameData {
	VkSemaphore _renderSemaphore, _presentSemaphore;
	VkFence _renderFence;

	VkCommandPool _commandPool;
	VkCommandBuffer _mainCommandBuffer;

	AllocatedBuffer cameraBuffer;
	VkDescriptorSet globalDescriptor;

	AllocatedBuffer objectBuffer;
	VkDescriptorSet objectDescriptor;
};

struct Material {
	VkPipeline pipeline;
	VkPipelineLayout pipelineLayout;
	VkDescriptorSet textureSet{VK_NULL_HANDLE}; //texture defaulted to null
};

struct RenderObject {
	Mesh* mesh;
	Material* material;
	glm::mat4 transformMatrix;
};

struct MeshPushConstants {
    glm::vec4 data;
    glm::mat4 render_matrix;
};

struct DeletionQueue {
    std::deque<std::function<void()>> deletors;

    void push_function(std::function<void()>&& function) {
        deletors.push_back(function);
    }

    void flush() {
        for (auto it = deletors.rbegin(); it != deletors.rend(); it++) {
            (*it)(); //call the function
        }
        deletors.clear();
    }
};

struct UploadContext {
    VkFence _uploadFence;
    VkCommandPool _commandPool;
};

struct Texture {
    AllocatedImage image;
    VkImageView imageView;
};

class PipelineBuilder {
public:
	std::vector<VkPipelineShaderStageCreateInfo> _shaderStages;
	VkPipelineVertexInputStateCreateInfo _vertexInputInfo;
	VkPipelineInputAssemblyStateCreateInfo _inputAssembly;
	VkViewport _viewport;
	VkRect2D _scissor;
	VkPipelineRasterizationStateCreateInfo _rasterizer;
	VkPipelineColorBlendAttachmentState _colorBlendAttachment;
	VkPipelineDepthStencilStateCreateInfo _depthStencil;
	VkPipelineMultisampleStateCreateInfo _multisampling;
	VkPipelineLayout _pipelineLayout;

	VkPipeline build_pipeline(VkDevice device, VkRenderPass pass);
};


class VulkanEngine {
public:

	bool _isInitialized{ false };
	int _frameNumber {0};

	VkExtent2D _windowExtent{ 1700 , 900 };

	struct SDL_Window* _window{ nullptr };

	// --- omitted ---
    VkInstance _instance; // Vulkan library handle
	VkDebugUtilsMessengerEXT _debug_messenger; // Vulkan debug output handle
	VkPhysicalDevice _chosenGPU; // GPU chosen as the default device
	VkDevice _device; // Vulkan device for commands
	VkSurfaceKHR _surface; // Vulkan window surface

	VkPhysicalDeviceProperties _gpuProperties;

	// --- other code ---
	VkSwapchainKHR _swapchain; // from other articles

	// image format expected by the windowing system
	VkFormat _swapchainImageFormat;

	//array of images from the swapchain
	std::vector<VkImage> _swapchainImages;

	//array of image-views from the swapchain
	std::vector<VkImageView> _swapchainImageViews;

	VkQueue _graphicsQueue;
	uint32_t _graphicsQueueFamily;

	VkRenderPass _renderPass;
	std::vector<VkFramebuffer> _framebuffers;

	FrameData _frames[FRAME_OVERLAP];

	VkDescriptorSetLayout _globalSetLayout;
	VkDescriptorSetLayout _objectSetLayout;
	VkDescriptorSetLayout _singleTextureSetLayout;
    VkDescriptorPool _descriptorPool;

	int _selectedShader{ 0 };

	DeletionQueue _mainDeletionQueue;

	VmaAllocator _allocator;

	AllocatedImage _depthImage;
	VkImageView _depthImageView;
	VkFormat _depthFormat;

	std::vector<RenderObject> _renderables;
	std::unordered_map<std::string, Mesh> _meshes;
	std::unordered_map<std::string, Material> _materials;
	std::unordered_map<std::string, Texture> _loadedTextures;

	GPUSceneData _sceneParameters;
    AllocatedBuffer _sceneParameterBuffer;


	UploadContext _uploadContext;
	void immediate_submit(std::function<void(VkCommandBuffer cmd)>&& function);

	FrameData& get_current_frame();

	Material* create_material(VkPipeline pipeline, VkPipelineLayout pipelineLayout, const std::string& name);
    Material* get_material(const std::string& name);

    Mesh* get_mesh(const std::string& name); 

    void draw_objects(VkCommandBuffer cmd, RenderObject* first, int count);

	AllocatedBuffer create_buffer(size_t allocSize, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage);
	size_t pad_uniform_buffer_size(size_t originalSize);

	//initializes everything in the engine
	void init();

	//shuts down the engine
	void cleanup();

	//draw loop
	void draw();

	//run main loop
	void run();

private:

	void init_vulkan();

	void init_swapchain();

	void init_commands();

	void init_default_renderpass();

	void init_framebuffers();

	void init_sync_structures();

	void init_pipelines();

	bool load_shader_module(const char* filePath, VkShaderModule* outShaderModule);

	void init_scene();

    void init_descriptors();

	void load_meshes();
	void upload_mesh(Mesh& mesh);

	void load_images();
};
