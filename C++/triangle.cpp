
// C++ app to draw a triangle, using wgpu-native (https://github.com/gfx-rs/wgpu-native)
// as the WebGPU implementation and GLFW (+ glfw3webgpu) for the window/surface,

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <GLFW/glfw3.h>
#include <glfw3webgpu.h>
#include <webgpu/webgpu.h>

namespace {

// mirrors wgpu.gpu.request_adapter_sync() / adapter.request_device_sync():
// wgpu-native's C API is asynchronous by design (it can also target the browser),
// but on native platforms the callback is invoked before the request_* function
// returns, so these small structs just let us read the result back out.
struct AdapterRequest {
    WGPUAdapter adapter = nullptr;
};

struct DeviceRequest {
    WGPUDevice device = nullptr;
};

void handle_request_adapter(WGPURequestAdapterStatus status, WGPUAdapter adapter,
                             WGPUStringView message, void *userdata1, void * /*userdata2*/) {
    if (status == WGPURequestAdapterStatus_Success) {
        static_cast<AdapterRequest *>(userdata1)->adapter = adapter;
    } else {
        std::fprintf(stderr, "request_adapter failed: %.*s\n", (int)message.length, message.data);
    }
}

void handle_request_device(WGPURequestDeviceStatus status, WGPUDevice device,
                            WGPUStringView message, void *userdata1, void * /*userdata2*/) {
    if (status == WGPURequestDeviceStatus_Success) {
        static_cast<DeviceRequest *>(userdata1)->device = device;
    } else {
        std::fprintf(stderr, "request_device failed: %.*s\n", (int)message.length, message.data);
    }
}

// mirrors device.create_buffer_with_data(data=..., usage=...) from wgpu-py:
// wgpu-native has no such helper, so a buffer is created already mapped
// (mappedAtCreation), the data is copied into the mapped range, and the
// buffer is unmapped so the GPU can use it.
WGPUBuffer create_buffer_with_data(WGPUDevice device, const void *data, size_t size,
                                    WGPUBufferUsage usage) {
    WGPUBufferDescriptor descriptor = {};
    descriptor.usage = usage;
    descriptor.size = size;
    descriptor.mappedAtCreation = true;

    WGPUBuffer buffer = wgpuDeviceCreateBuffer(device, &descriptor);
    void *mapped_range = wgpuBufferGetMappedRange(buffer, 0, size);
    std::memcpy(mapped_range, data, size);
    wgpuBufferUnmap(buffer);
    return buffer;
}

} // namespace

int main() {
    // 1. Window/canvas
    // GLFW plays the role of rendercanvas here: it creates and owns the native
    // window, drives the event loop, and (via glfw3webgpu) creates the
    // WGPUSurface used to present frames.
    //   update_mode "ondemand" | "continuous" | "manual" in rendercanvas has no
    //   direct GLFW equivalent; "ondemand" (redraw only when requested) is
    //   reproduced below with a needs_redraw flag and glfwWaitEvents(), instead
    //   of polling and redrawing on every loop iteration.
    if (!glfwInit()) {
        std::fprintf(stderr, "failed to initialize GLFW\n");
        std::exit(EXIT_FAILURE);
    }

    // wgpu-native talks directly to the native windowing system, not to any
    // particular graphics API, so tell GLFW not to set one up itself.
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow *window = glfwCreateWindow(640, 480, "WebGPU triangle", nullptr, nullptr);
    if (!window) {
        std::fprintf(stderr, "failed to create window\n");
        std::exit(EXIT_FAILURE);
    }

    // create a WebGPU instance, and get a surface for the window from it:
    // together these play the role of canvas.get_context("wgpu") in wgpu-py.
    WGPUInstance instance = wgpuCreateInstance(nullptr);
    WGPUSurface surface = glfwCreateWindowWGPUSurface(instance, window);

    // 2. Adapter and device
    // adapter is the interface to the GPU: integrated, discrete, or virtual GPU (SW).
    // request a GPU adapter with high-performance preference and compatible with the surface
    WGPURequestAdapterOptions adapter_options = {};
    adapter_options.powerPreference = WGPUPowerPreference_HighPerformance;
    adapter_options.compatibleSurface = surface;

    AdapterRequest adapter_request;
    WGPURequestAdapterCallbackInfo adapter_callback_info = {};
    adapter_callback_info.mode = WGPUCallbackMode_AllowProcessEvents;
    adapter_callback_info.callback = handle_request_adapter;
    adapter_callback_info.userdata1 = &adapter_request;
    wgpuInstanceRequestAdapter(instance, &adapter_options, adapter_callback_info);
    WGPUAdapter adapter = adapter_request.adapter;

    // device is the interface to the GPU, and is used to create resources and submit work
    // request a device from the adapter
    DeviceRequest device_request;
    WGPURequestDeviceCallbackInfo device_callback_info = {};
    device_callback_info.mode = WGPUCallbackMode_AllowProcessEvents;
    device_callback_info.callback = handle_request_device;
    device_callback_info.userdata1 = &device_request;
    wgpuAdapterRequestDevice(adapter, nullptr, device_callback_info);
    WGPUDevice device = device_request.device;

    // 3. Configure canvas
    // retrieve the preferred texture format for the adapter and configure the surface with the device and format
    // wgpu-native has no context.get_preferred_format(); the equivalent call is
    // wgpuSurfaceGetCapabilities(), whose formats[0] is the preferred format.
    WGPUSurfaceCapabilities surface_capabilities = {};
    wgpuSurfaceGetCapabilities(surface, adapter, &surface_capabilities);
    WGPUTextureFormat texture_format = surface_capabilities.formats[0]; // e.g. BGRA8Unorm, RGBA8UnormSrgb, RGBA16Float

    // configure the surface with the device and format
    // effect of alpha_mode: surface composting with the window background, or not
    //   premultiplied: Cout = Cs + Cd x (1 - As), Aout = As + Ad x (1 - As)
    //   unpremultiplied: Cout = Cs x As + Cd x (1 - As), Aout = As + Ad x (1 - As)
    //   opaque: Cout = Cs, Aout = 1
    // surface composition is a final image composition; it takes place after blending
    WGPUSurfaceConfiguration surface_config = {};
    surface_config.device = device;
    surface_config.format = texture_format;
    surface_config.usage = WGPUTextureUsage_RenderAttachment;
    surface_config.presentMode = WGPUPresentMode_Fifo;
    surface_config.alphaMode = WGPUCompositeAlphaMode_Opaque;
    {
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        surface_config.width = static_cast<uint32_t>(width);
        surface_config.height = static_cast<uint32_t>(height);
    }
    wgpuSurfaceConfigure(surface, &surface_config);

    // 4. Vertex data: coordinates and colors
    // std::array is contiguous binary data suitable for uploading to a GPU buffer
    std::array<float, 6> coords = {
         0.0f,  0.7f,
        -0.7f, -0.7f,
         0.7f, -0.7f,
    };
    std::array<uint8_t, 12> colors = {
        255, 0, 0, 255,
        0, 255, 0, 255,
        0, 0, 255, 255,
    };

    // GPU buffers are used to store data on the GPU, and can be used as vertex buffers, index buffers, uniform buffers, storage buffers, etc.
    // create a GPU buffer for the vertex coordinates and colors
    //   possible usage flags:
    //    WGPUBufferUsage_Vertex: for vertex buffers
    //    WGPUBufferUsage_Index: for index buffers
    //    WGPUBufferUsage_Uniform: for uniform buffers
    //    WGPUBufferUsage_Storage: for storage buffers
    //   that can be combined with:
    //    WGPUBufferUsage_CopySrc: for copying data from the buffer
    //    WGPUBufferUsage_CopyDst: for copying data to the buffer
    WGPUBuffer coord_buffer =
        create_buffer_with_data(device, coords.data(), coords.size() * sizeof(float), WGPUBufferUsage_Vertex);
    WGPUBuffer color_buffer =
        create_buffer_with_data(device, colors.data(), colors.size() * sizeof(uint8_t), WGPUBufferUsage_Vertex);

    // 5. Shaders
    // Shaders are programs that run on the GPU, and are used to process vertex data and fragment data.
    // WGSL (WebGPU Shading Language) is a shading language for WebGPU, similar to GLSL or HLSL.
    const char *shader_code = R"(
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec3<f32>,
};

@vertex
fn vertex_main (
      @location(0) position: vec2<f32>,
      @location(1) color: vec3<f32>,
) -> VertexOutput {
    var output: VertexOutput;
    output.position = vec4<f32>(position, 0.0, 1.0);
    output.color = color;
    return output;
}

@fragment
fn fragment_main (input: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(input.color, 1.0);
}
)";

    WGPUShaderSourceWGSL wgsl_source = {};
    wgsl_source.chain.sType = WGPUSType_ShaderSourceWGSL;
    wgsl_source.code = {shader_code, WGPU_STRLEN};

    WGPUShaderModuleDescriptor shader_descriptor = {};
    shader_descriptor.nextInChain = &wgsl_source.chain;
    WGPUShaderModule shader = wgpuDeviceCreateShaderModule(device, &shader_descriptor);

    // 6. Pipeline (immutable)

    // A pipeline is a configured GPU program describing how a draw or compute operation works.
    // A render pipeline contains:
    //   Vertex and fragment shaders
    //   Vertex-buffer layouts
    //   Primitive type: triangles, lines, points
    //   Rasterization and face-culling settings
    //   Color formats and blending
    //   Depth/stencil and multisampling settings
    //   Resource-binding layout

    // Pipeline layout describes the resources (buffers, textures, samplers) that are used by the shaders.
    // wgpu-py can derive this automatically from the shader (layout="auto"); the
    // wgpu-native C API has no such helper, so we build the layout explicitly.
    // Since this pipeline uses no bind groups at all, an empty layout is the
    // exact equivalent of the derived "auto" layout here.
    //   There are two ways to create a pipeline layout:
    //   1. Automatic (wgpu-py only): wgpu derives layouts from shader declarations. Convenient for simple pipelines.
    //   2. Explicitly create a pipeline layout with bind group layouts, and use it.
    //      Useful for sharing bind groups between pipelines and controlling compatibility.
    WGPUPipelineLayoutDescriptor pipeline_layout_descriptor = {};
    WGPUPipelineLayout pipeline_layout = wgpuDeviceCreatePipelineLayout(device, &pipeline_layout_descriptor);

    WGPUVertexAttribute coord_attribute = {};
    coord_attribute.shaderLocation = 0;
    coord_attribute.offset = 0;
    coord_attribute.format = WGPUVertexFormat_Float32x2;

    WGPUVertexAttribute color_attribute = {};
    color_attribute.shaderLocation = 1;
    color_attribute.offset = 0;
    color_attribute.format = WGPUVertexFormat_Unorm8x4; // normalized unsigned byte values

    std::array<WGPUVertexBufferLayout, 2> vertex_buffer_layouts = {};
    vertex_buffer_layouts[0].arrayStride = 2 * 4; // two float32 values
    vertex_buffer_layouts[0].stepMode = WGPUVertexStepMode_Vertex;
    vertex_buffer_layouts[0].attributeCount = 1;
    vertex_buffer_layouts[0].attributes = &coord_attribute;

    vertex_buffer_layouts[1].arrayStride = 4 * 1; // four unsigned byte values: stride must be multiple of 4 bytes for alignment
    vertex_buffer_layouts[1].stepMode = WGPUVertexStepMode_Vertex;
    vertex_buffer_layouts[1].attributeCount = 1;
    vertex_buffer_layouts[1].attributes = &color_attribute;

    WGPUColorTargetState color_target = {};
    color_target.format = texture_format;
    color_target.writeMask = WGPUColorWriteMask_All;

    WGPUFragmentState fragment_state = {};
    fragment_state.module = shader;
    fragment_state.entryPoint = {"fragment_main", WGPU_STRLEN};
    fragment_state.targetCount = 1;
    fragment_state.targets = &color_target;

    WGPURenderPipelineDescriptor pipeline_descriptor = {};
    pipeline_descriptor.layout = pipeline_layout;
    pipeline_descriptor.vertex.module = shader;
    pipeline_descriptor.vertex.entryPoint = {"vertex_main", WGPU_STRLEN};
    pipeline_descriptor.vertex.bufferCount = vertex_buffer_layouts.size();
    pipeline_descriptor.vertex.buffers = vertex_buffer_layouts.data();
    pipeline_descriptor.primitive.topology = WGPUPrimitiveTopology_TriangleList;
    pipeline_descriptor.primitive.frontFace = WGPUFrontFace_CCW;
    pipeline_descriptor.primitive.cullMode = WGPUCullMode_None;
    pipeline_descriptor.multisample.count = 1;
    pipeline_descriptor.multisample.mask = 0xFFFFFFFF;
    pipeline_descriptor.fragment = &fragment_state;

    WGPURenderPipeline pipeline = wgpuDeviceCreateRenderPipeline(device, &pipeline_descriptor);

    // 7. Event handling
    // Callbacks for keyboard and mouse events can be registered with the window.
    glfwSetKeyCallback(window, [](GLFWwindow * /*window*/, int key, int /*scancode*/, int action, int /*mods*/) {
        if (action == GLFW_PRESS) {
            std::printf("%d\n", key);
        }
    });

    glfwSetMouseButtonCallback(window, [](GLFWwindow *window, int button, int action, int /*mods*/) {
        if (action == GLFW_PRESS) {
            double x, y;
            glfwGetCursorPos(window, &x, &y);
            std::printf("%f %f %d\n", x, y, button);
        }
    });

    // 8. Render one frame

    // To render a frame, we need to:
    //   1. Create a command encoder
    //   2. Begin a render pass
    //   3. Set the pipeline and vertex buffers
    //   4. Draw the vertices
    //   5. End the render pass
    //   6. Submit the command buffer to the queue

    WGPUQueue queue = wgpuDeviceGetQueue(device);

    // Renders one frame with raw wgpu calls: begins a render pass on the
    // surface's current texture, binds the pipeline/vertex buffers, draws the
    // 3 vertices, and submits. Called from the main loop below whenever a
    // redraw is needed.
    auto draw = [&]() {
        WGPUSurfaceTexture surface_texture = {};
        wgpuSurfaceGetCurrentTexture(surface, &surface_texture);
        WGPUTextureView view = wgpuTextureCreateView(surface_texture.texture, nullptr);

        // Encoder and render pass are one-use objects, and must be created for each frame.
        WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device, nullptr);

        // A render pass is a sequence of rendering commands that are executed together, and can be used to render to one or more textures.
        WGPURenderPassColorAttachment color_attachment = {};
        color_attachment.view = view; // the texture view to render to
        color_attachment.depthSlice = WGPU_DEPTH_SLICE_UNDEFINED;
        color_attachment.loadOp = WGPULoadOp_Clear;   // Clear to clear the texture, Load to keep the previous contents
        color_attachment.storeOp = WGPUStoreOp_Store; // Store to keep the contents after the pass, Discard to discard the contents
        color_attachment.clearValue = {1.0, 1.0, 1.0, 1.0}; // white background

        WGPURenderPassDescriptor render_pass_descriptor = {};
        render_pass_descriptor.colorAttachmentCount = 1;
        render_pass_descriptor.colorAttachments = &color_attachment;

        WGPURenderPassEncoder render_pass = wgpuCommandEncoderBeginRenderPass(encoder, &render_pass_descriptor);

        wgpuRenderPassEncoderSetPipeline(render_pass, pipeline);
        // set the vertex buffers: first buffer is at slot 0, second buffer is at slot 1
        wgpuRenderPassEncoderSetVertexBuffer(render_pass, 0, coord_buffer, 0, WGPU_WHOLE_SIZE);
        wgpuRenderPassEncoderSetVertexBuffer(render_pass, 1, color_buffer, 0, WGPU_WHOLE_SIZE);
        wgpuRenderPassEncoderDraw(render_pass, 3, 1, 0, 0); // indicates the number of vertices to draw, starting from vertex 0
        wgpuRenderPassEncoderEnd(render_pass); // end the render pass
        wgpuRenderPassEncoderRelease(render_pass);

        // complete the command encoder
        WGPUCommandBuffer command_buffer = wgpuCommandEncoderFinish(encoder, nullptr);

        // a queue is used to sends work and data from the CPU to the GPU: commands as executed in order
        // each command buffer can be submitted only once
        // execution is asynchronous, and the CPU can continue to run while the GPU is working
        wgpuQueueSubmit(queue, 1, &command_buffer); // submit the command buffer to the queue
        wgpuSurfacePresent(surface);

        wgpuCommandBufferRelease(command_buffer);
        wgpuCommandEncoderRelease(encoder);
        wgpuTextureViewRelease(view);
        wgpuTextureRelease(surface_texture.texture);
    };

    // set the draw function to be called when the window needs to be redrawn;
    // this plays the role of canvas.request_draw(draw) in wgpu-py.
    bool needs_redraw = true;
    glfwSetWindowUserPointer(window, &needs_redraw);
    glfwSetWindowRefreshCallback(window, [](GLFWwindow *window) {
        *static_cast<bool *>(glfwGetWindowUserPointer(window)) = true;
    });
    glfwSetFramebufferSizeCallback(window, [](GLFWwindow *window, int width, int height) {
        if (width == 0 || height == 0) {
            return;
        }
        *static_cast<bool *>(glfwGetWindowUserPointer(window)) = true;
    });

    // start the GLFW event loop: process events and render frames
    // with "continuous" update mode, the draw function would be called continuously at max_fps
    // with "ondemand" update mode (reproduced here), the draw function is called only when requested
    while (!glfwWindowShouldClose(window)) {
        glfwWaitEvents();

        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        if (width == 0 || height == 0) {
            continue;
        }
        if (static_cast<uint32_t>(width) != surface_config.width ||
            static_cast<uint32_t>(height) != surface_config.height) {
            surface_config.width = static_cast<uint32_t>(width);
            surface_config.height = static_cast<uint32_t>(height);
            wgpuSurfaceConfigure(surface, &surface_config);
            needs_redraw = true;
        }

        if (needs_redraw) {
            needs_redraw = false;
            draw();
        }
    }

    wgpuRenderPipelineRelease(pipeline);
    wgpuPipelineLayoutRelease(pipeline_layout);
    wgpuShaderModuleRelease(shader);
    wgpuBufferRelease(color_buffer);
    wgpuBufferRelease(coord_buffer);
    wgpuSurfaceCapabilitiesFreeMembers(surface_capabilities);
    wgpuQueueRelease(queue);
    wgpuDeviceRelease(device);
    wgpuAdapterRelease(adapter);
    wgpuSurfaceRelease(surface);
    glfwDestroyWindow(window);
    wgpuInstanceRelease(instance);
    glfwTerminate();
    return 0;
}
