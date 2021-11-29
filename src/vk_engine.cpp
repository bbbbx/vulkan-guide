
#include "vk_engine.h"

#include <SDL.h>
#include <SDL_vulkan.h>

#include <vk_types.h>
#include <vk_initializers.h>
#include "vk_textures.h"

#include "VkBootstrap.h"
#include <iostream>
#include <fstream>

#define VMA_IMPLEMENTATION
#include "vk_mem_alloc.h"

#include <glm/gtx/transform.hpp>

using namespace std;
#define VK_CHECK(x)                                                     \
	do {                                                                \
		VkResult err = x;                                               \
		if (err) {                                                      \
			std::cout << "Detected Vulkan error: " << err << std::endl; \
			abort();                                                    \
		}                                                               \
	} while(0)

void VulkanEngine::init()
{
	// We initialize SDL and create a window with it. 
	SDL_Init(SDL_INIT_VIDEO);

	SDL_WindowFlags window_flags = (SDL_WindowFlags)(SDL_WINDOW_VULKAN);
	
	_window = SDL_CreateWindow(
		"Vulkan Engine",
		SDL_WINDOWPOS_UNDEFINED,
		SDL_WINDOWPOS_UNDEFINED,
		_windowExtent.width,
		_windowExtent.height,
		window_flags
	);
	
	//load the core Vulkan structures
	init_vulkan();

	init_swapchain();

	init_commands();

	init_default_renderpass();

	init_framebuffers();

	init_sync_structures();

	init_descriptors();
	
	init_pipelines();

	load_images();

	load_meshes();

	init_scene();

	//everything went fine
	_isInitialized = true;
}

void VulkanEngine::init_vulkan()
{
	vkb::InstanceBuilder builder;

	//make the Vulkan instance, with basic debug features
	unsigned int pCount;
	SDL_Vulkan_GetInstanceExtensions(_window, &pCount, nullptr);
	std::vector<const char*> pNames(pCount);
	SDL_Vulkan_GetInstanceExtensions(_window, &pCount, pNames.data());
	builder.set_app_name("Example Vulkan Application")
		.request_validation_layers(true)
		.require_api_version(1, 1, 0)
		.use_default_debug_messenger();
	for (size_t i = 0; i < pCount; i++)
	{
		builder.enable_extension(pNames[i]);
	}

	auto inst_ret = builder.build();

	if (!inst_ret) {
        std::cerr << "Failed to create Vulkan instance. Error: " << inst_ret.error().message() << "\n";
        return;
    }
	vkb::Instance vkb_inst = inst_ret.value();

	//store the instance
	_instance = vkb_inst.instance;
	//store the debug messenger
	_debug_messenger = vkb_inst.debug_messenger;

	// get the surface of the window we opened with SDL
	if (SDL_Vulkan_CreateSurface(_window, _instance, &_surface) == SDL_FALSE) {
		throw std::runtime_error("failed to create surface!");
	}

	// for gl_BaseInstance
	// see https://www.reddit.com/r/vulkan/comments/or8e8u/vkcreateshadermodule_runtime_validation_error/
	// for Vulkan 1.0
	// VkPhysicalDeviceVulkan11Features vk11Features = {};
	// vk11Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
	// vk11Features.pNext = NULL;
	// vk11Features.shaderDrawParameters = VK_TRUE;
	// for Vulkan 1.1
	VkPhysicalDeviceShaderDrawParametersFeatures drawParametersFeatures;
	drawParametersFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_DRAW_PARAMETERS_FEATURES;
	drawParametersFeatures.pNext = nullptr;
	drawParametersFeatures.shaderDrawParameters = VK_TRUE;
	// for Vulkan 1.2, it is core
	// https://vulkan.lunarg.com/doc/view/1.2.189.0/mac/1.2-extensions/vkspec.html#spirvenv-capabilities-table-DrawParameters
	// Specification is your best friend!


	//use vkbootstrap to select a GPU.
	//We want a GPU that can write to the SDL surface and supports Vulkan 1.1
	vkb::PhysicalDeviceSelector selector{ vkb_inst };
	std::vector<const char*> extensions{
		VK_KHR_SHADER_DRAW_PARAMETERS_EXTENSION_NAME,
#if defined(__APPLE__)
		"VK_KHR_portability_subset"
#endif
	};
	selector.add_required_extensions(extensions);
	auto phys_ret = selector
		.set_minimum_version(1, 1)
		.set_surface(_surface)
		.select();

	if (!phys_ret) {
        std::cerr << "Failed to select Vulkan Physical Device. Error: " << phys_ret.error().message() << "\n";
        return;
    }
	//create the final Vulkan device
	vkb::DeviceBuilder deviceBuilder{ phys_ret.value() };
	auto dev_ret = deviceBuilder.build();
	deviceBuilder.add_pNext(&drawParametersFeatures);
	if (!dev_ret) {
        std::cerr << "Failed to create Vulkan device. Error: " << dev_ret.error().message() << "\n";
        return;
    }
	vkb::Device vkbDevice = dev_ret.value ();

	// Get the VkDevice handle used in the rest of a Vulkan application
	_device = vkbDevice.device;
	_chosenGPU = phys_ret.value().physical_device;

	vkGetPhysicalDeviceProperties(_chosenGPU, &_gpuProperties);
	std::cout << "The GPU has minimum buffer alignment of " << _gpuProperties.limits.minUniformBufferOffsetAlignment << std::endl;

	_graphicsQueue = vkbDevice.get_queue(vkb::QueueType::graphics).value();
	_graphicsQueueFamily = vkbDevice.get_queue_index(vkb::QueueType::graphics).value();


	VmaAllocatorCreateInfo vmaallocInfo = {};
	vmaallocInfo.instance = _instance;
	vmaallocInfo.physicalDevice = _chosenGPU;
	vmaallocInfo.device = _device;
	vmaCreateAllocator(&vmaallocInfo, &_allocator);
}

void VulkanEngine::init_swapchain()
{
	vkb::SwapchainBuilder swapchainBuilder{_chosenGPU,_device,_surface };

	vkb::Swapchain vkbSwapchain = swapchainBuilder
		.use_default_format_selection()
		//use vsync present mode
		.set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
		.set_desired_extent(_windowExtent.width, _windowExtent.height)
		.build()
		.value();

	//store swapchain and its related images
	_swapchain = vkbSwapchain.swapchain;
	_swapchainImages = vkbSwapchain.get_images().value();
	_swapchainImageViews = vkbSwapchain.get_image_views().value();

	_swapchainImageFormat = vkbSwapchain.image_format;

	_mainDeletionQueue.push_function([=]() {
        vkDestroySwapchainKHR(_device, _swapchain, nullptr);
    });

	VkExtent3D depthImageExtent = {
        _windowExtent.width,
        _windowExtent.height,
        1
    };

	_depthFormat = VK_FORMAT_D32_SFLOAT;

	VkImageCreateInfo dimg_info = vkinit::image_create_info(_depthFormat, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, depthImageExtent);
    VmaAllocationCreateInfo dimg_allocinfo = {};
    dimg_allocinfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
    dimg_allocinfo.requiredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    vmaCreateImage(_allocator, &dimg_info, &dimg_allocinfo, &_depthImage._image, &_depthImage._allocation, nullptr);

    VkImageViewCreateInfo dview_info = vkinit::imageview_create_info(_depthFormat, _depthImage._image, VK_IMAGE_ASPECT_DEPTH_BIT);
    VK_CHECK(vkCreateImageView(_device, &dview_info, nullptr, &_depthImageView));

	_mainDeletionQueue.push_function([=]() {
        vkDestroyImageView(_device, _depthImageView, nullptr);
        vmaDestroyImage(_allocator, _depthImage._image,  _depthImage._allocation);
    });
}

void VulkanEngine::init_commands() {
    VkCommandPoolCreateInfo commandPoolInfo = vkinit::command_pool_create_info(_graphicsQueueFamily, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);

    for (int i = 0; i < FRAME_OVERLAP; i++) {
        VK_CHECK(vkCreateCommandPool(_device, &commandPoolInfo, nullptr, &_frames[i]._commandPool));

        VkCommandBufferAllocateInfo allocInfo = vkinit::command_buffer_allocate_info(_frames[i]._commandPool, 1);
        VK_CHECK(vkAllocateCommandBuffers(_device, &allocInfo, &_frames[i]._mainCommandBuffer));

        _mainDeletionQueue.push_function([=]() {
            vkDestroyCommandPool(_device, _frames[i]._commandPool, nullptr);
        });
    }

	VkCommandPoolCreateInfo uploadCommandPoolInfo = vkinit::command_pool_create_info(_graphicsQueueFamily);

	VK_CHECK(vkCreateCommandPool(_device, &uploadCommandPoolInfo, nullptr, &_uploadContext._commandPool));

	_mainDeletionQueue.push_function([=]() {
		vkDestroyCommandPool(_device, _uploadContext._commandPool, nullptr);
	});
}

void VulkanEngine::init_default_renderpass() {
	VkAttachmentDescription color_attachment = {};
	//the attachment will have the format needed by the swapchain
	color_attachment.format = _swapchainImageFormat;
	color_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
	color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	color_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	color_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;

	//we don't know or care about the starting layout of the attachment
	color_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	//after the renderpass ends, the image has to be on a layout ready for display
	color_attachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

	VkAttachmentReference color_attachment_ref = {};
	//attachment number will index into the pAttachments array in the parent renderpass itself
	color_attachment_ref.attachment = 0;
	color_attachment_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

	VkAttachmentDescription depth_attachment = {};
    depth_attachment.flags = 0;
    depth_attachment.format = _depthFormat;
    depth_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
    depth_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depth_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    depth_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depth_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depth_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    depth_attachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference depth_attachment_ref = {};
    depth_attachment_ref.attachment = 1;
    depth_attachment_ref.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

	VkSubpassDescription subpass = {};
	subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
	subpass.colorAttachmentCount = 1;
	subpass.pColorAttachments = &color_attachment_ref;
	subpass.pDepthStencilAttachment = &depth_attachment_ref;

    VkAttachmentDescription attachments[2] = { color_attachment, depth_attachment };

	VkRenderPassCreateInfo render_pass_info = {};
	render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
	render_pass_info.attachmentCount = 2;
	render_pass_info.pAttachments = &attachments[0];
	render_pass_info.subpassCount = 1;
	render_pass_info.pSubpasses = &subpass;

	VK_CHECK(vkCreateRenderPass(_device, &render_pass_info, nullptr, &_renderPass));

	_mainDeletionQueue.push_function([=]() {
        vkDestroyRenderPass(_device, _renderPass, nullptr);
    });
}

void VulkanEngine::init_framebuffers() {
	VkFramebufferCreateInfo fb_info = {};
	fb_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
	fb_info.pNext = nullptr;
	fb_info.renderPass = _renderPass;
	fb_info.width = _windowExtent.width;
	fb_info.height = _windowExtent.height;
	fb_info.layers = 1;

	const size_t swapchain_imagecount = _swapchainImages.size();
	_framebuffers = std::vector<VkFramebuffer>(swapchain_imagecount);

	for (size_t i = 0; i < swapchain_imagecount; i++)
	{
		VkImageView attachments[2];
		attachments[0] = _swapchainImageViews[i];
		attachments[1] = _depthImageView;

		fb_info.attachmentCount = 2;
		fb_info.pAttachments = attachments;
		VK_CHECK(vkCreateFramebuffer(_device, &fb_info, nullptr, &_framebuffers[i]));

		_mainDeletionQueue.push_function([=]() {
            vkDestroyFramebuffer(_device, _framebuffers[i], nullptr);
            vkDestroyImageView(_device, _swapchainImageViews[i], nullptr);
        });
	}
}

void VulkanEngine::init_sync_structures() {
    VkFenceCreateInfo fenceCreateInfo = vkinit::fence_create_info(VK_FENCE_CREATE_SIGNALED_BIT);

    VkSemaphoreCreateInfo semaphoreCreateInfo = vkinit::semaphore_create_info();

	for (int i = 0; i < FRAME_OVERLAP; i++) {
        VK_CHECK(vkCreateFence(_device, &fenceCreateInfo, nullptr, &_frames[i]._renderFence));

        _mainDeletionQueue.push_function([=]() {
            vkDestroyFence(_device, _frames[i]._renderFence, nullptr);
        });

        VK_CHECK(vkCreateSemaphore(_device, &semaphoreCreateInfo, nullptr, &_frames[i]._renderSemaphore));
        VK_CHECK(vkCreateSemaphore(_device, &semaphoreCreateInfo, nullptr, &_frames[i]._presentSemaphore));

        _mainDeletionQueue.push_function([=]() {
            vkDestroySemaphore(_device, _frames[i]._renderSemaphore, nullptr);
            vkDestroySemaphore(_device, _frames[i]._presentSemaphore, nullptr);
        });
    }

	VkFenceCreateInfo uploadFenceCreateInfo = vkinit::fence_create_info();

	VK_CHECK(vkCreateFence(_device, &uploadFenceCreateInfo, nullptr, &_uploadContext._uploadFence));

	_mainDeletionQueue.push_function([=]() {
		vkDestroyFence(_device, _uploadContext._uploadFence, nullptr);
	});

}

void VulkanEngine::init_pipelines() {
	VkShaderModule colorMeshShader, meshVertShader;
	if (!load_shader_module("../shaders/default_lit.frag.spv", &colorMeshShader)) {
		std::cout << "Error when building the colored mesh shader" << std::endl;
	} else {
		std::cout << "Colored Triangle fragment shader successfully loaded" << std::endl;
	}

    if (!load_shader_module("../shaders/tri_mesh_descriptors.vert.spv", &meshVertShader)) {
        std::cout << "Error when building the triangle vertex shader module" << std::endl;
    } else {
        std::cout << "Vertex shader successfully loaded" << std::endl;
    }


	VkShaderModule texturedMeshVertShader, texturedMeshFragShader;
	if (!load_shader_module("../shaders/tri_mesh_textured.vert.spv", &texturedMeshVertShader)) {
        std::cout << "Error when building the triangle vertex shader module" << std::endl;
    } else {
        std::cout << "Textured mesh vertex shader successfully loaded" << std::endl;
    }
    if (!load_shader_module("../shaders/textured_lit.frag.spv", &texturedMeshFragShader)) {
        std::cout << "Error when building the textured mesh shader" << std::endl;
    } else {
        std::cout << "Textured mesh fragment shader successfully loaded" << std::endl;
    }

	VkPipelineLayoutCreateInfo mesh_pipeline_layout_create_info = vkinit::pipeline_layout_create_info();

	VkPushConstantRange push_constant;
	push_constant.offset = 0;
	push_constant.size = sizeof(MeshPushConstants);
	push_constant.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

	mesh_pipeline_layout_create_info.pushConstantRangeCount = 1;
	mesh_pipeline_layout_create_info.pPushConstantRanges = &push_constant;

	mesh_pipeline_layout_create_info.setLayoutCount = 2;
	VkDescriptorSetLayout setLayouts[] = { _globalSetLayout, _objectSetLayout };
	mesh_pipeline_layout_create_info.pSetLayouts = setLayouts;

	VkPipelineLayout meshPipelineLayout;
	VK_CHECK(vkCreatePipelineLayout(_device, &mesh_pipeline_layout_create_info, nullptr, &meshPipelineLayout));


	PipelineBuilder pipelineBuilder;
    pipelineBuilder._shaderStages.push_back(
        vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_VERTEX_BIT, meshVertShader));
    pipelineBuilder._shaderStages.push_back(
        vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_FRAGMENT_BIT, colorMeshShader));

	pipelineBuilder._vertexInputInfo = vkinit::vertex_input_state_create_info();
	pipelineBuilder._inputAssembly = vkinit::input_assembly_create_info(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);

	pipelineBuilder._viewport.x = 0.0f;
	pipelineBuilder._viewport.y = 0.0f;
	pipelineBuilder._viewport.width = (float)_windowExtent.width;
	pipelineBuilder._viewport.height = (float)_windowExtent.height;
	pipelineBuilder._viewport.minDepth = 0.0f;
	pipelineBuilder._viewport.maxDepth = 1.0f;

	pipelineBuilder._scissor.offset = { 0, 0 };
	pipelineBuilder._scissor.extent = _windowExtent;

	pipelineBuilder._rasterizer = vkinit::rasterization_state_create_info(VK_POLYGON_MODE_FILL);

	pipelineBuilder._multisampling = vkinit::multisampling_state_create_info();

	pipelineBuilder._colorBlendAttachment = vkinit::color_blend_attachment_state();

	pipelineBuilder._depthStencil = vkinit::pipeline_depth_stencil_state_create_info(true, true, VK_COMPARE_OP_LESS_OR_EQUAL);

	VertexInputDescription description = Vertex::get_vertex_description();

    pipelineBuilder._vertexInputInfo.vertexBindingDescriptionCount = description.bindings.size();
    pipelineBuilder._vertexInputInfo.pVertexBindingDescriptions = description.bindings.data();

    pipelineBuilder._vertexInputInfo.vertexAttributeDescriptionCount = description.attributes.size();
    pipelineBuilder._vertexInputInfo.pVertexAttributeDescriptions = description.attributes.data();

	pipelineBuilder._pipelineLayout = meshPipelineLayout;

    VkPipeline meshPipeline = pipelineBuilder.build_pipeline(_device, _renderPass);


	create_material(meshPipeline, meshPipelineLayout, "defaultmesh");

	VkPipelineLayoutCreateInfo textured_pipeline_layout_info = mesh_pipeline_layout_create_info;

    VkDescriptorSetLayout texturedSetLayouts[] = { _globalSetLayout, _objectSetLayout, _singleTextureSetLayout };

    textured_pipeline_layout_info.setLayoutCount = 3;
    textured_pipeline_layout_info.pSetLayouts = texturedSetLayouts;

    VkPipelineLayout texturedPipelineLayout;
    VK_CHECK(vkCreatePipelineLayout(_device, &textured_pipeline_layout_info, nullptr, &texturedPipelineLayout));

	//create pipeline for textured drawing
    pipelineBuilder._shaderStages.clear();
    pipelineBuilder._shaderStages.push_back(
        vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_VERTEX_BIT, texturedMeshVertShader));
    pipelineBuilder._shaderStages.push_back(
        vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_FRAGMENT_BIT, texturedMeshFragShader));

	pipelineBuilder._pipelineLayout = texturedPipelineLayout;
    VkPipeline texPipeline = pipelineBuilder.build_pipeline(_device, _renderPass);
    create_material(texPipeline, texturedPipelineLayout, "texturedmesh");


    vkDestroyShaderModule(_device, colorMeshShader, nullptr);
    vkDestroyShaderModule(_device, meshVertShader, nullptr);
    vkDestroyShaderModule(_device, texturedMeshVertShader, nullptr);
    vkDestroyShaderModule(_device, texturedMeshFragShader, nullptr);
    _mainDeletionQueue.push_function([=]() {
        vkDestroyPipeline(_device, meshPipeline, nullptr);
        vkDestroyPipeline(_device, texPipeline, nullptr);

		vkDestroyPipelineLayout(_device, meshPipelineLayout, nullptr);
		vkDestroyPipelineLayout(_device, texturedPipelineLayout, nullptr);
    });
}

bool VulkanEngine::load_shader_module(const char* filePath, VkShaderModule* outShaderModule) {
	std::ifstream file(filePath, std::ios::ate | std::ios::binary);

	if (!file.is_open()) {
		return false;
	}

	size_t fileSize = (size_t)file.tellg();

	//spirv expects the buffer to be on uint32, so make sure to reserve an int vector big enough for the entire file
	std::vector<uint32_t> buffer(fileSize / sizeof(uint32_t));

	//put file cursor at beginning
	file.seekg(0);

	//load the entire file into the buffer
	file.read((char*)buffer.data(), fileSize);

	file.close();

	VkShaderModuleCreateInfo createInfo = {};
	createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	createInfo.pNext = nullptr;
	createInfo.codeSize = buffer.size() * sizeof(uint32_t);
	createInfo.pCode = buffer.data();

	VkShaderModule shaderModule;
	if (vkCreateShaderModule(_device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
		return false;
	}
	*outShaderModule = shaderModule;
	return true;
}

void VulkanEngine::cleanup()
{	
	if (_isInitialized) {

		//make sure the gpu has stopped doing its things
		vkDeviceWaitIdle(_device);

		_mainDeletionQueue.flush();

		vmaDestroyAllocator(_allocator);

		vkDestroySurfaceKHR(_instance, _surface, nullptr);

		vkDestroyDevice(_device, nullptr);
		vkb::destroy_debug_utils_messenger(_instance, _debug_messenger);
		vkDestroyInstance(_instance, nullptr);

		SDL_DestroyWindow(_window);
	}
}

void VulkanEngine::draw()
{
	//nothing yet
	//wait until the GPU has finished rendering the last frame. Timeout of 1 second
	VK_CHECK(vkWaitForFences(_device, 1, &get_current_frame()._renderFence, true, 1000000000));
	VK_CHECK(vkResetFences(_device, 1, &get_current_frame()._renderFence));

	VK_CHECK(vkResetCommandBuffer(get_current_frame()._mainCommandBuffer, 0));

	//request image from the swapchain, one second timeout
	uint32_t swapChainImageIndex;
	VK_CHECK(vkAcquireNextImageKHR(_device, _swapchain, 1000000000, get_current_frame()._presentSemaphore, nullptr, &swapChainImageIndex));

	VkCommandBuffer cmd = get_current_frame()._mainCommandBuffer;

    VkCommandBufferBeginInfo cmdBeginInfo = vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

	VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBeginInfo));

	VkClearValue clearValue;
	float flash = abs( sin(_frameNumber / 120.0f) );
	clearValue.color = { { 0.0f, 0.0f, flash, 1.0f } };

	VkClearValue depthClear;
	depthClear.depthStencil.depth = 1.0f;

	VkRenderPassBeginInfo rpInfo = vkinit::renderpass_begin_info(_renderPass, _windowExtent, _framebuffers[swapChainImageIndex]);
	rpInfo.clearValueCount = 2;
	VkClearValue clearValues[] = { clearValue, depthClear };
	rpInfo.pClearValues = clearValues;

	vkCmdBeginRenderPass(cmd, &rpInfo, VK_SUBPASS_CONTENTS_INLINE);

	draw_objects(cmd, _renderables.data(), _renderables.size());

	vkCmdEndRenderPass(cmd);

	VK_CHECK(vkEndCommandBuffer(cmd));

	VkSubmitInfo submit = {};
	submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submit.pNext = nullptr;

	VkPipelineStageFlags waitStages = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	submit.pWaitDstStageMask = &waitStages;

	submit.waitSemaphoreCount = 1;
	submit.pWaitSemaphores = &get_current_frame()._presentSemaphore;

	submit.signalSemaphoreCount = 1;
	submit.pSignalSemaphores = &get_current_frame()._renderSemaphore;

	submit.commandBufferCount = 1;
	submit.pCommandBuffers = &cmd;

	VK_CHECK(vkQueueSubmit(_graphicsQueue, 1, &submit, get_current_frame()._renderFence));

	//present
	VkPresentInfoKHR presentInfo = {};
	presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
	presentInfo.pNext = nullptr;

	presentInfo.swapchainCount = 1;
	presentInfo.pSwapchains = &_swapchain;

	presentInfo.waitSemaphoreCount = 1;
	presentInfo.pWaitSemaphores = &get_current_frame()._renderSemaphore;

	presentInfo.pImageIndices = &swapChainImageIndex;

	VK_CHECK(vkQueuePresentKHR(_graphicsQueue, &presentInfo));

	_frameNumber++;

}

void VulkanEngine::run()
{
	SDL_Event e;
	bool bQuit = false;

	//main loop
	while (!bQuit)
	{
		//Handle events on queue
		while (SDL_PollEvent(&e) != 0)
		{
			//close the window when user alt-f4s or clicks the X button			
			if (e.type == SDL_QUIT) {
				bQuit = true;
			} else if (e.type == SDL_KEYDOWN) {
				if (e.key.keysym.sym == SDLK_SPACE) {
					_selectedShader++;
					if (_selectedShader > 1) {
						_selectedShader = 0;
					}
				}
			}
		}

		draw();
	}
}



VkPipeline PipelineBuilder::build_pipeline(VkDevice device, VkRenderPass pass) {
	VkPipelineViewportStateCreateInfo viewportState = {};
	viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
	viewportState.pNext = nullptr;

	viewportState.viewportCount = 1;
	viewportState.pViewports = &_viewport;
	viewportState.scissorCount = 1;
	viewportState.pScissors = &_scissor;


	VkPipelineColorBlendStateCreateInfo colorBlending = {};
	colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
	colorBlending.pNext = nullptr;

	colorBlending.logicOpEnable = VK_FALSE;
	colorBlending.logicOp = VK_LOGIC_OP_COPY;
	colorBlending.attachmentCount = 1;
	colorBlending.pAttachments = &_colorBlendAttachment;


	VkGraphicsPipelineCreateInfo pipelineInfo = {};
	pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
	pipelineInfo.pNext = nullptr;

	pipelineInfo.stageCount = _shaderStages.size();
	pipelineInfo.pStages = _shaderStages.data();
	pipelineInfo.pVertexInputState = &_vertexInputInfo;
	pipelineInfo.pInputAssemblyState = &_inputAssembly;
	pipelineInfo.pViewportState = &viewportState;
	pipelineInfo.pRasterizationState = &_rasterizer;
	pipelineInfo.pMultisampleState = &_multisampling;
	pipelineInfo.pColorBlendState = &colorBlending;
	pipelineInfo.layout = _pipelineLayout;
	pipelineInfo.renderPass = pass;
	pipelineInfo.subpass = 0;
	pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;

	pipelineInfo.pDepthStencilState = &_depthStencil;

	VkPipeline newPipeline;
	if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &newPipeline) != VK_SUCCESS) {
		throw std::runtime_error("failed to create pipeline!");
		return VK_NULL_HANDLE;
	}

	return newPipeline;
}

void VulkanEngine::load_meshes() {
	Mesh triangleMesh;
	triangleMesh._vertices.resize(3);

    triangleMesh._vertices[0].position = { 1.0f, 1.0f, 0.0f };
    triangleMesh._vertices[1].position = {-1.0f, 1.0f, 0.0f };
    triangleMesh._vertices[2].position = { 0.0f,-1.0f, 0.0f };

    triangleMesh._vertices[0].color = { 0.0f, 1.0f, 0.0f };
    triangleMesh._vertices[1].color = { 0.0f, 1.0f, 0.0f };
    triangleMesh._vertices[2].color = { 0.0f, 1.0f, 0.0f };

	Mesh monkeyMesh;
	monkeyMesh.load_from_obj("../assets/monkey_smooth.obj");

	Mesh roomMesh;
	roomMesh.load_from_obj("../assets/viking_room.obj");

    upload_mesh(triangleMesh);
    upload_mesh(monkeyMesh);
    upload_mesh(roomMesh);

	_meshes["triangle"] = triangleMesh;
    _meshes["monkey"] = monkeyMesh;
    _meshes["room"] = roomMesh;


	Mesh lostEmpire{};
    lostEmpire.load_from_obj("../assets/lost_empire.obj");

    upload_mesh(lostEmpire);

    _meshes["empire"] = lostEmpire;
}

void VulkanEngine::upload_mesh(Mesh& mesh) {
	const size_t bufferSize = sizeof(Vertex) * mesh._vertices.size();

	VkBufferCreateInfo stagingBufferInfo = {};
    stagingBufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	stagingBufferInfo.pNext = nullptr;
    stagingBufferInfo.size = bufferSize;
    stagingBufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

    VmaAllocationCreateInfo vmaallocInfo = {};
    vmaallocInfo.usage = VMA_MEMORY_USAGE_CPU_ONLY;

	AllocatedBuffer stagingBuffer;

    VK_CHECK(vmaCreateBuffer(_allocator, &stagingBufferInfo, &vmaallocInfo,
        &stagingBuffer._buffer,
        &stagingBuffer._allocation,
        nullptr));

	void* data;
    vmaMapMemory(_allocator, stagingBuffer._allocation, &data);

    memcpy(data, mesh._vertices.data(), bufferSize);

    vmaUnmapMemory(_allocator, stagingBuffer._allocation);

	VkBufferCreateInfo vertexBufferInfo = {};
	vertexBufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	vertexBufferInfo.pNext = nullptr;
    vertexBufferInfo.size = bufferSize;
    vertexBufferInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

	vmaallocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

	VK_CHECK(vmaCreateBuffer(_allocator, &vertexBufferInfo, &vmaallocInfo,
        &mesh._vertexBuffer._buffer,
        &mesh._vertexBuffer._allocation,
        nullptr));

	immediate_submit([=](VkCommandBuffer cmd) {
        VkBufferCopy copy;
        copy.srcOffset = 0;
        copy.dstOffset = 0;
        copy.size = bufferSize;
        vkCmdCopyBuffer(cmd, stagingBuffer._buffer, mesh._vertexBuffer._buffer, 1, &copy);
    });

	_mainDeletionQueue.push_function([=]() {
        vmaDestroyBuffer(_allocator, mesh._vertexBuffer._buffer, mesh._vertexBuffer._allocation);
    });

	vmaDestroyBuffer(_allocator, stagingBuffer._buffer, stagingBuffer._allocation);
}

void VulkanEngine::init_scene() {
	// add monkey model
    RenderObject monkey;
    monkey.mesh = get_mesh("monkey");
    monkey.material = get_material("defaultmesh");
	glm::mat4 translation = glm::translate(glm::mat4{ 1.0f }, glm::vec3(0.0f, 3.0f, 0.0f));
    monkey.transformMatrix = translation;

    _renderables.push_back(monkey);

	// add room model
	RenderObject room;
    room.mesh = get_mesh("room");
    room.material = get_material("defaultmesh");
	glm::mat4 rotation = glm::rotate(glm::radians(-90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
	glm::mat4 scaling = glm::scale(glm::vec3(2.5f));
	glm::mat4 roomTranslation = glm::translate(glm::mat4{ 1.0f }, glm::vec3(0.0f, 5.0f, 0.0f));
	room.transformMatrix = roomTranslation * scaling * rotation;

    _renderables.push_back(room);

	RenderObject map;
    map.mesh = get_mesh("empire");
    map.material = get_material("texturedmesh");
    map.transformMatrix = glm::translate(glm::vec3{5, -10, 0});

    _renderables.push_back(map);

	// add triangles
    for (int x = -30; x <= 30; x++) {
        for (int y = -30; y <= 30; y++) {
            RenderObject tri;
            tri.mesh = get_mesh("triangle");
            tri.material = get_material("defaultmesh");

            glm::mat4 translation = glm::translate(glm::mat4{ 1.0f }, glm::vec3(x, -1.0f, y));
            glm::mat4 scale = glm::scale(glm::mat4{ 1.0f }, glm::vec3(0.2, 0.2, 0.2));
            tri.transformMatrix = translation * scale;

            _renderables.push_back(tri);
        }
    }


	VkSamplerCreateInfo samplerInfo = vkinit::sampler_create_info(VK_FILTER_NEAREST);

    VkSampler blockySampler;
    vkCreateSampler(_device, &samplerInfo, nullptr, &blockySampler);

    Material* texturedMat = get_material("texturedmesh");

    VkDescriptorSetAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.pNext = nullptr;
    allocInfo.descriptorPool = _descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &_singleTextureSetLayout;

    vkAllocateDescriptorSets(_device, &allocInfo, &texturedMat->textureSet);

    VkDescriptorImageInfo imageBufferInfo;
    imageBufferInfo.sampler = blockySampler;
    imageBufferInfo.imageView = _loadedTextures["empire_diffuse"].imageView;
    imageBufferInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkWriteDescriptorSet texture1 = vkinit::write_descriptor_image(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, texturedMat->textureSet, &imageBufferInfo, 0);

    vkUpdateDescriptorSets(_device, 1, &texture1, 0, nullptr);

	_mainDeletionQueue.push_function([=]() {
		vkDestroySampler(_device, blockySampler, nullptr);
	});
}

Material* VulkanEngine::create_material(VkPipeline pipeline, VkPipelineLayout pipelineLayout, const std::string& name) {
    Material mat;
    mat.pipeline = pipeline;
    mat.pipelineLayout = pipelineLayout;
    _materials[name] = mat;
    return &_materials[name];
}

Material* VulkanEngine::get_material(const std::string& name) {
    auto it = _materials.find(name);
    if (it == _materials.end()) {
        return nullptr;
    } else {
        return &(*it).second;
    }
}

Mesh* VulkanEngine::get_mesh(const std::string& name) {
    auto it = _meshes.find(name);
    if (it == _meshes.end()) {
        return nullptr;
    } else {
        return &(*it).second;
    }
}

void VulkanEngine::draw_objects(VkCommandBuffer cmd, RenderObject* first, int count) {
	float x = cos(_frameNumber / 120.0f) * 7.0f;
	float z = sin(_frameNumber / 120.0f) * 7.0f;
	glm::vec3 camPos = { x, 5.0f, z };
	glm::mat4 view = glm::lookAt(camPos, glm::vec3(0.0f, 4.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));

    glm::mat4 projection = glm::perspective(glm::radians(70.0f), 1700.0f / 900.0f, 0.1f, 200.0f);
    projection[1][1] *= -1;

	GPUCameraData camData;
    camData.view = view;
    camData.projection = projection;
    camData.viewproj = projection * view;

	void* data;
    vmaMapMemory(_allocator, get_current_frame().cameraBuffer._allocation, &data);

    memcpy(data, &camData, sizeof(GPUCameraData));

    vmaUnmapMemory(_allocator, get_current_frame().cameraBuffer._allocation);

	float framed = _frameNumber / 120.0f;
	_sceneParameters.ambientColor = { sin(framed), 0, cos(framed), 1 };

	char* sceneData;
	vmaMapMemory(_allocator, _sceneParameterBuffer._allocation, (void**)&sceneData);

	int frameIndex = _frameNumber % FRAME_OVERLAP;
	sceneData += pad_uniform_buffer_size(sizeof(GPUSceneData)) * frameIndex;

	memcpy(sceneData, &_sceneParameters, sizeof(GPUSceneData));

	vmaUnmapMemory(_allocator, _sceneParameterBuffer._allocation);

	void* objectData;
	vmaMapMemory(_allocator, get_current_frame().objectBuffer._allocation, &objectData);

	GPUObjectData* objectSSBO = (GPUObjectData*)objectData;

	for (int i = 0; i < count; i++) {
		RenderObject& object = first[i];
		objectSSBO[i].modelMatrix = object.transformMatrix;
	}

	vmaUnmapMemory(_allocator, get_current_frame().objectBuffer._allocation);

    Mesh* lastMesh = nullptr;
    Material* lastMaterial = nullptr;

    for (int i = 0; i < count; i++) {
        RenderObject& object = first[i];

        //only bind the pipeline if it doesn't match with the already bound one
        if (object.material != lastMaterial) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, object.material->pipeline);
            lastMaterial = object.material;

			//offset for our scene buffer
			uint32_t uniform_offset = pad_uniform_buffer_size(sizeof(GPUSceneData)) * frameIndex;
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, object.material->pipelineLayout, 0, 1, &get_current_frame().globalDescriptor, 1, &uniform_offset);

			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, object.material->pipelineLayout, 1, 1, &get_current_frame().objectDescriptor, 0, nullptr);

			if (object.material->textureSet != VK_NULL_HANDLE) {
				//texture descriptor
				vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, object.material->pipelineLayout, 2, 1, &object.material->textureSet, 0, nullptr);
			}
        }

        glm::mat4 model = object.transformMatrix;
        glm::mat4 mesh_matrix = projection * view * model;

        MeshPushConstants push_constants;
        push_constants.render_matrix = model;

        vkCmdPushConstants(cmd, object.material->pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(MeshPushConstants), &push_constants);

        //only bind the mesh if it's a different one from last bind
        if (object.mesh != lastMesh) {
            VkDeviceSize offset = 0;
            vkCmdBindVertexBuffers(cmd, 0, 1, &object.mesh->_vertexBuffer._buffer, &offset);
            lastMesh = object.mesh;
        }

        //we can now draw
        vkCmdDraw(cmd, object.mesh->_vertices.size(), 1, 0, i);
    }
}

FrameData& VulkanEngine::get_current_frame() {
	return _frames[_frameNumber % FRAME_OVERLAP];
}

AllocatedBuffer VulkanEngine::create_buffer(size_t allocSize, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage)
{
	//allocate vertex buffer
	VkBufferCreateInfo bufferInfo = {};
	bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	bufferInfo.pNext = nullptr;

	bufferInfo.size = allocSize;
	bufferInfo.usage = usage;


	VmaAllocationCreateInfo vmaallocInfo = {};
	vmaallocInfo.usage = memoryUsage;

	AllocatedBuffer newBuffer;

	//allocate the buffer
	VK_CHECK(vmaCreateBuffer(_allocator, &bufferInfo, &vmaallocInfo,
		&newBuffer._buffer,
		&newBuffer._allocation,
		nullptr));

	return newBuffer;
}

void VulkanEngine::init_descriptors() {
	//binding for camera data at 0
	VkDescriptorSetLayoutBinding camBufferBinding = vkinit::descriptor_layout_binding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT, 0);
	//binding for scene data at 1
	VkDescriptorSetLayoutBinding sceneBufferBinding = vkinit::descriptor_layout_binding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 1);

    VkDescriptorSetLayoutCreateInfo setInfo = {};
    setInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    setInfo.pNext = nullptr;

	VkDescriptorSetLayoutBinding bindings[] = { camBufferBinding, sceneBufferBinding };
    setInfo.bindingCount = 2;
    setInfo.pBindings = bindings;
    //no flags
    setInfo.flags = 0;

    vkCreateDescriptorSetLayout(_device, &setInfo, nullptr, &_globalSetLayout);


    VkDescriptorSetLayoutBinding objectBind = vkinit::descriptor_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_VERTEX_BIT, 0);

    VkDescriptorSetLayoutCreateInfo set2Info = {};
	set2Info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    set2Info.pNext = nullptr;
    set2Info.bindingCount = 1;
    set2Info.pBindings = &objectBind;
    set2Info.flags = 0;

	vkCreateDescriptorSetLayout(_device, &set2Info, nullptr, &_objectSetLayout);


	VkDescriptorSetLayoutBinding textureBind = vkinit::descriptor_layout_binding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 0);

    VkDescriptorSetLayoutCreateInfo set3Info = {};
    set3Info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    set3Info.pNext = nullptr;
    set3Info.bindingCount = 1;
    set3Info.pBindings = &textureBind;
    set3Info.flags = 0;

    vkCreateDescriptorSetLayout(_device, &set3Info, nullptr, &_singleTextureSetLayout);

	//create a descriptor pool that will hold 10 uniform buffers, 10 dynamic uniform buffers and 10 storage buffers
	std::vector<VkDescriptorPoolSize> sizes = {
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 10 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 10 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 10 },
		//add combined-image-sampler descriptor types to the pool
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 10 }
    };

	VkDescriptorPoolCreateInfo pool_info = {};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    
    pool_info.maxSets = 10;
    pool_info.poolSizeCount = (uint32_t)sizes.size();
    pool_info.pPoolSizes = sizes.data();
    pool_info.flags = 0;

    vkCreateDescriptorPool(_device, &pool_info, nullptr, &_descriptorPool);

	const size_t sceneParamBufferSize = FRAME_OVERLAP * pad_uniform_buffer_size(sizeof(GPUSceneData));
	_sceneParameterBuffer = create_buffer(sceneParamBufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

    for (int i = 0; i < FRAME_OVERLAP; i++) {
        _frames[i].cameraBuffer = create_buffer(sizeof(GPUCameraData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

		const int MAX_OBJECTS = 10000;
		_frames[i].objectBuffer = create_buffer(sizeof(GPUObjectData) * MAX_OBJECTS, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

		VkDescriptorSetAllocateInfo allocInfo = {};
		allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		allocInfo.pNext = nullptr;

		allocInfo.descriptorPool = _descriptorPool;
		// 只要分配 1 个 descriptor set
		allocInfo.descriptorSetCount = 1;

		// 多个 layout？可以为不同的 descriptor sets 指定不同的 layout？
		allocInfo.pSetLayouts = &_globalSetLayout;

		vkAllocateDescriptorSets(_device, &allocInfo, &_frames[i].globalDescriptor);


		//allocate the descriptor set that will point to object buffer
        VkDescriptorSetAllocateInfo objectSetAlloc = {};
        objectSetAlloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        objectSetAlloc.pNext = nullptr;
        objectSetAlloc.descriptorPool = _descriptorPool;
        objectSetAlloc.descriptorSetCount = 1;
        objectSetAlloc.pSetLayouts = &_objectSetLayout;

        vkAllocateDescriptorSets(_device, &objectSetAlloc, &_frames[i].objectDescriptor);


		//information about the buffer we want to point at in the descriptor
		VkDescriptorBufferInfo cameraInfo = {};
		cameraInfo.buffer = _frames[i].cameraBuffer._buffer;
		cameraInfo.offset = 0;
		cameraInfo.range = sizeof(GPUCameraData);

		VkDescriptorBufferInfo sceneInfo;
		sceneInfo.buffer = _sceneParameterBuffer._buffer;
		sceneInfo.offset = 0;
		sceneInfo.range = sizeof(GPUSceneData);

		VkDescriptorBufferInfo objectBufferInfo;
        objectBufferInfo.buffer = _frames[i].objectBuffer._buffer;
        objectBufferInfo.offset = 0;
        objectBufferInfo.range = sizeof(GPUObjectData) * MAX_OBJECTS;

		VkWriteDescriptorSet cameraWrite = vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, _frames[i].globalDescriptor, &cameraInfo, 0);
		VkWriteDescriptorSet sceneWrite = vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, _frames[i].globalDescriptor, &sceneInfo, 1);
		VkWriteDescriptorSet objectWrite = vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, _frames[i].objectDescriptor, &objectBufferInfo, 0);

		VkWriteDescriptorSet setWrites[] = { cameraWrite, sceneWrite, objectWrite };
		vkUpdateDescriptorSets(_device, 3, setWrites, 0, nullptr);
    }

	_mainDeletionQueue.push_function([=]() {
		vkDestroyDescriptorSetLayout(_device, _globalSetLayout, nullptr);
		vkDestroyDescriptorSetLayout(_device, _objectSetLayout, nullptr);
		vkDestroyDescriptorSetLayout(_device, _singleTextureSetLayout, nullptr);
		vkDestroyDescriptorPool(_device, _descriptorPool, nullptr);

		vmaDestroyBuffer(_allocator, _sceneParameterBuffer._buffer, _sceneParameterBuffer._allocation);

		for (int i = 0; i < FRAME_OVERLAP; i++) {
			vmaDestroyBuffer(_allocator, _frames[i].cameraBuffer._buffer, _frames[i].cameraBuffer._allocation);
			vmaDestroyBuffer(_allocator, _frames[i].objectBuffer._buffer, _frames[i].objectBuffer._allocation);
		}
	});
}

size_t VulkanEngine::pad_uniform_buffer_size(size_t originalSize) {
    size_t minUboAlignment = _gpuProperties.limits.minUniformBufferOffsetAlignment;
    size_t alignedSize = originalSize;
    if (minUboAlignment > 0) {
        // alignedSize += minUboAlignment - (originalSize % minUboAlignment);
        alignedSize = (alignedSize + minUboAlignment - 1) & ~(minUboAlignment - 1);
    }
    return alignedSize;
}

void VulkanEngine::immediate_submit(std::function<void(VkCommandBuffer cmd)>&& function) {
    VkCommandBufferAllocateInfo cmdAllocInfo = vkinit::command_buffer_allocate_info(_uploadContext._commandPool, 1);

    VkCommandBuffer cmd;
    VK_CHECK(vkAllocateCommandBuffers(_device, &cmdAllocInfo, &cmd));

    //begin the command buffer recording. We will use this command buffer exactly once, so we want to let vulkan know that
    VkCommandBufferBeginInfo cmdBeginInfo = vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
    VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBeginInfo));

    //execute the function
    function(cmd);

    VK_CHECK(vkEndCommandBuffer(cmd));

    VkSubmitInfo submit = vkinit::submit_info(&cmd);

    //submit command buffer to the queue and execute it.
    // _uploadFence will now block until the graphic commands finish execution
    VK_CHECK(vkQueueSubmit(_graphicsQueue, 1, &submit, _uploadContext._uploadFence));
    
    vkWaitForFences(_device, 1, &_uploadContext._uploadFence, true, 9999999999);
    vkResetFences(_device, 1, &_uploadContext._uploadFence);

    //clear the command pool. This will free the command buffer too
    vkResetCommandPool(_device, _uploadContext._commandPool, 0);
}

void VulkanEngine::load_images() {
    Texture lostEmpire;

    vkutil::load_image_from_file(*this, "../assets/lost_empire-RGBA.png", lostEmpire.image);

    VkImageViewCreateInfo imageinfo = vkinit::imageview_create_info(VK_FORMAT_R8G8B8A8_SRGB, lostEmpire.image._image, VK_IMAGE_ASPECT_COLOR_BIT);
    vkCreateImageView(_device, &imageinfo, nullptr, &lostEmpire.imageView);

    _loadedTextures["empire_diffuse"] = lostEmpire;

	_mainDeletionQueue.push_function([=]() {
		vkDestroyImageView(_device, lostEmpire.imageView, nullptr);
	});
}