
#include "vk_engine.h"

#include <SDL.h>
#include <SDL_vulkan.h>

#include <vk_types.h>
#include <vk_initializers.h>

#include "VkBootstrap.h"
#include <iostream>
#include <fstream>

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
	
	init_pipelines();

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


	//use vkbootstrap to select a GPU.
	//We want a GPU that can write to the SDL surface and supports Vulkan 1.1
	vkb::PhysicalDeviceSelector selector{ vkb_inst };
	std::vector<const char*> extensions{
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
	if (!dev_ret) {
        std::cerr << "Failed to create Vulkan device. Error: " << dev_ret.error().message() << "\n";
        return;
    }
	vkb::Device vkbDevice = dev_ret.value ();

	// Get the VkDevice handle used in the rest of a Vulkan application
	_device = vkbDevice.device;
	_chosenGPU = phys_ret.value().physical_device;

	_graphicsQueue = vkbDevice.get_queue(vkb::QueueType::graphics).value();
	_graphicsQueueFamily = vkbDevice.get_queue_index(vkb::QueueType::graphics).value();
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

}

void VulkanEngine::init_commands() {
	VkCommandPoolCreateInfo commandPoolInfo = vkinit::command_pool_create_info(_graphicsQueueFamily, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);
	VK_CHECK(vkCreateCommandPool(_device, &commandPoolInfo, nullptr, &_commandPool));

	_mainDeletionQueue.push_function([=]() {
		vkDestroyCommandPool(_device, _commandPool, nullptr);
	});

	VkCommandBufferAllocateInfo commandBufferAllocateInfo = vkinit::command_buffer_allocate_info(_commandPool, 1, VK_COMMAND_BUFFER_LEVEL_PRIMARY);
	VK_CHECK(vkAllocateCommandBuffers(_device, &commandBufferAllocateInfo, &_mainCommandBuffer));
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

	VkSubpassDescription subpass = {};
	subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
	subpass.colorAttachmentCount = 1;
	subpass.pColorAttachments = &color_attachment_ref;

	VkRenderPassCreateInfo render_pass_info = {};
	render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
	render_pass_info.attachmentCount = 1;
	render_pass_info.pAttachments = &color_attachment;
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
	fb_info.attachmentCount = 1;
	fb_info.width = _windowExtent.width;
	fb_info.height = _windowExtent.height;
	fb_info.layers = 1;

	const size_t swapchain_imagecount = _swapchainImages.size();
	_framebuffers = std::vector<VkFramebuffer>(swapchain_imagecount);

	for (size_t i = 0; i < swapchain_imagecount; i++)
	{
		fb_info.pAttachments = &_swapchainImageViews[i];
		VK_CHECK(vkCreateFramebuffer(_device, &fb_info, nullptr, &_framebuffers[i]));

		_mainDeletionQueue.push_function([=]() {
            vkDestroyFramebuffer(_device, _framebuffers[i], nullptr);
            vkDestroyImageView(_device, _swapchainImageViews[i], nullptr);
        });
	}
}

void VulkanEngine::init_sync_structures() {
	VkFenceCreateInfo fenceCreateInfo = {};
	fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
	fenceCreateInfo.pNext = nullptr;

	//we want to create the fence with the Create Signaled flag, so we can wait on it before using it on a GPU command (for the first frame)
	fenceCreateInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

	VK_CHECK(vkCreateFence(_device, &fenceCreateInfo, nullptr, &_renderFence));

	_mainDeletionQueue.push_function([=]() {
        vkDestroyFence(_device, _renderFence, nullptr);
    });


	VkSemaphoreCreateInfo semaphoreCreateInfo = {};
	semaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
	semaphoreCreateInfo.pNext = nullptr;
	//for the semaphores we don't need any flags
	semaphoreCreateInfo.flags = 0;

	VK_CHECK(vkCreateSemaphore(_device, &semaphoreCreateInfo, nullptr, &_presentSemaphore));
	VK_CHECK(vkCreateSemaphore(_device, &semaphoreCreateInfo, nullptr, &_renderSemaphore));

	_mainDeletionQueue.push_function([=]() {
        vkDestroySemaphore(_device, _renderSemaphore, nullptr);
        vkDestroySemaphore(_device, _presentSemaphore, nullptr);
    });
}

void VulkanEngine::init_pipelines() {
	if (!load_shader_module("../shaders/colored_triangle.frag.spv", &triangleFragShader)) {
		throw std::runtime_error("Error when building the triangle fragment shader module");
	} else {
		std::cout << "Colored Triangle fragment shader successfully loaded" << std::endl;
	}

	if (!load_shader_module("../shaders/colored_triangle.vert.spv", &triangleVertShader)) {
		throw std::runtime_error("Error when building the triangle vertex shader module");
	} else {
		std::cout << "Colored Triangle vertex shader successfully loaded" << std::endl;
	}

	VkPipelineLayoutCreateInfo pipeline_layout_info = vkinit::pipeline_layout_create_info();
	VK_CHECK(vkCreatePipelineLayout(_device, &pipeline_layout_info, nullptr, &_trianglePipelineLayout));

	PipelineBuilder pipelineBuilder;
	pipelineBuilder._shaderStages.push_back(
		vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_VERTEX_BIT, triangleVertShader)
	);
	pipelineBuilder._shaderStages.push_back(
		vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_FRAGMENT_BIT, triangleFragShader)
	);

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

	pipelineBuilder._pipelineLayout = _trianglePipelineLayout;

	_trianglePipeline = pipelineBuilder.build_pipeline(_device, _renderPass);	




	VkShaderModule redTriangleVertShader, redTriangleFragShader;
	if (!load_shader_module("../shaders/triangle.vert.spv", &redTriangleVertShader)) {
		throw std::runtime_error("Error when building the triangle vertex shader module");
	} else {
		std::cout << "Triangle vertex shader successfully loaded" << std::endl;
	}
	if (!load_shader_module("../shaders/triangle.frag.spv", &redTriangleFragShader)) {
		throw std::runtime_error("Error when building the triangle fragment shader module");
	} else {
		std::cout << "Triangle fragment shader successfully loaded" << std::endl;
	}
	pipelineBuilder._shaderStages.clear();

	pipelineBuilder._shaderStages.push_back(
		vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_VERTEX_BIT, redTriangleVertShader)
	);
	pipelineBuilder._shaderStages.push_back(
		vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_FRAGMENT_BIT, redTriangleFragShader)
	);
	
	_redTrianglePipeline = pipelineBuilder.build_pipeline(_device, _renderPass);
	

	vkDestroyShaderModule(_device, redTriangleVertShader, nullptr);
    vkDestroyShaderModule(_device, redTriangleFragShader, nullptr);
    vkDestroyShaderModule(_device, triangleFragShader, nullptr);
	vkDestroyShaderModule(_device, triangleVertShader, nullptr);

    _mainDeletionQueue.push_function([=]() {
        vkDestroyPipeline(_device, _trianglePipeline, nullptr);
        vkDestroyPipeline(_device, _redTrianglePipeline, nullptr);

        vkDestroyPipelineLayout(_device, _trianglePipelineLayout, nullptr);
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
		vkWaitForFences(_device, 1, &_renderFence, true, 1000000000);

		_mainDeletionQueue.flush();


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
	VK_CHECK(vkWaitForFences(_device, 1, &_renderFence, true, 1000000000));
	VK_CHECK(vkResetFences(_device, 1, &_renderFence));

	//request image from the swapchain, one second timeout
	uint32_t swapChainImageIndex;
	VK_CHECK(vkAcquireNextImageKHR(_device, _swapchain, 1000000000, _presentSemaphore, nullptr, &swapChainImageIndex));

	VkCommandBuffer cmd = _mainCommandBuffer;

	VkCommandBufferBeginInfo cmdBeginInfo = {};
	cmdBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	cmdBeginInfo.pNext = nullptr;

	cmdBeginInfo.pInheritanceInfo = nullptr;
	cmdBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

	VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBeginInfo));

	VkClearValue clearValue;
	float flash = abs( sin(_frameNumber / 120.0f) );
	clearValue.color = { { 0.0f, 0.0f, flash, 1.0f } };

	VkRenderPassBeginInfo rpInfo = {};
	rpInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
	rpInfo.pNext = nullptr;

	rpInfo.renderPass = _renderPass;
	rpInfo.renderArea.offset.x = 0;
	rpInfo.renderArea.offset.y = 0;
	rpInfo.renderArea.extent = _windowExtent;
	rpInfo.framebuffer = _framebuffers[swapChainImageIndex];

	rpInfo.clearValueCount = 1;
	rpInfo.pClearValues = &clearValue;

	vkCmdBeginRenderPass(cmd, &rpInfo, VK_SUBPASS_CONTENTS_INLINE);

	// 
	if (_selectedShader == 0) {
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _trianglePipeline);
	} else {
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _redTrianglePipeline);
	}
	vkCmdDraw(cmd, 3, 1, 0, 0);

	vkCmdEndRenderPass(cmd);

	VK_CHECK(vkEndCommandBuffer(cmd));

	VkSubmitInfo submit = {};
	submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submit.pNext = nullptr;

	VkPipelineStageFlags waitStages = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	submit.pWaitDstStageMask = &waitStages;

	submit.waitSemaphoreCount = 1;
	submit.pWaitSemaphores = &_presentSemaphore;

	submit.signalSemaphoreCount = 1;
	submit.pSignalSemaphores = &_renderSemaphore;

	submit.commandBufferCount = 1;
	submit.pCommandBuffers = &cmd;

	VK_CHECK(vkQueueSubmit(_graphicsQueue, 1, &submit, _renderFence));

	//present
	VkPresentInfoKHR presentInfo = {};
	presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
	presentInfo.pNext = nullptr;

	presentInfo.swapchainCount = 1;
	presentInfo.pSwapchains = &_swapchain;

	presentInfo.waitSemaphoreCount = 1;
	presentInfo.pWaitSemaphores = &_renderSemaphore;

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

	VkPipeline newPipeline;
	if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &newPipeline) != VK_SUCCESS) {
		throw std::runtime_error("failed to create pipeline!");
		return VK_NULL_HANDLE;
	}

	return newPipeline;
}