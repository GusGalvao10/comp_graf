// JavaScript port of triangle.py, using the browser's native WebGPU API
// (navigator.gpu / GPUCanvasContext) directly instead of Python's wgpu-py
// and rendercanvas -- the browser is the "windowing library" here.

async function main() {
    // 1. Window/canvas
    // RenderCanvas(
    //    size = (width, height),
    //    title ="Canvas title",
    //    update_mode = "ondemand" | "continuous" | "manual", ,
    //       ondemand: for static scene
    //           redraw only when requested (default): canvas.request_draw(), or canvas.request_redraw(), or to maintain min_fps
    //       continuous: for animation
    //           redraw continuously at max_fps
    //       manual: for debug or custom loops
    //           redraw only when requested: canvas.force_draw() or canvas.force_redraw()
    //    min_fps = 0.0,
    //    max_fps = 30.0,
    //    vsync = True,
    //    present_method = None | "screen" | "bitmap",
    //)
    // in the browser, the "canvas" is a real <canvas> element; there is no
    // update_mode to choose, since a static scene is simply drawn once, on
    // demand -- the same thing rendercanvas's default ("ondemand") does.
    const canvas = document.createElement("canvas");
    canvas.width = 640;
    canvas.height = 480;
    document.title = "WebGPU triangle";
    document.body.appendChild(canvas);

    // get the WebGPU context from the canvas
    const context = canvas.getContext("webgpu");

    // 2. Adapter and device
    // adapter is the interface to the GPU: integrated, discrete, or virtual GPU (SW).
    // request a GPU adapter with high-performance preference and compatible with the context
    const adapter = await navigator.gpu.requestAdapter({
        powerPreference: "high-performance",
    });
    if (!adapter) {
        throw new Error("no suitable GPU adapter found");
    }

    // device is the interface to the GPU, and is used to create resources and submit work
    // request a device from the adapter
    const device = await adapter.requestDevice();

    // 3. Configure canvas
    // retrieve the preferred texture format for the adapter and configure the context with the device and format
    const textureFormat = navigator.gpu.getPreferredCanvasFormat(); // e.g. "bgra8unorm", "rgba8unorm-srgb", "rgba16float"

    // configure the context with the device and format
    // effect of alpha_mode: canvas composting with the window background, or not
    //   premultiplied: Cout = Cs + Cd x (1 - As), Aout = As + Ad x (1 - As)
    //   opaque: Cout = Cs, Aout = 1
    // canvas composition is a final image composition; it takes place after blending
    context.configure({
        device: device,
        format: textureFormat,
        alphaMode: "opaque",
    });

    // 4. Vertex data: coordinates and colors
    // typed arrays are contiguous binary data suitable for uploading to a GPU buffer
    const coords = new Float32Array([
         0.0,  0.7,
        -0.7, -0.7,
         0.7, -0.7,
    ]);
    const colors = new Uint8Array([
        255, 0, 0, 255,
        0, 255, 0, 255,
        0, 0, 255, 255,
    ]);

    // GPU buffers are used to store data on the GPU, and can be used as vertex buffers, index buffers, uniform buffers, storage buffers, etc.
    // create a GPU buffer for the vertex coordinates and colors
    //   possible usage flags:
    //    GPUBufferUsage.VERTEX: for vertex buffers
    //    GPUBufferUsage.INDEX: for index buffers
    //    GPUBufferUsage.UNIFORM: for uniform buffers
    //    GPUBufferUsage.STORAGE: for storage buffers
    //   that can be combined with:
    //    GPUBufferUsage.COPY_SRC: for copying data from the buffer
    //    GPUBufferUsage.COPY_DST: for copying data to the buffer
    function createBufferWithData(data, usage) {
        const buffer = device.createBuffer({
            size: data.byteLength,
            usage: usage,
            mappedAtCreation: true,
        });
        new Uint8Array(buffer.getMappedRange()).set(
            new Uint8Array(data.buffer, data.byteOffset, data.byteLength)
        );
        buffer.unmap();
        return buffer;
    }

    const coordBuffer = createBufferWithData(coords, GPUBufferUsage.VERTEX);
    const colorBuffer = createBufferWithData(colors, GPUBufferUsage.VERTEX);

    // 5. Shaders
    // Shaders are programs that run on the GPU, and are used to process vertex data and fragment data.
    // WGSL (WebGPU Shading Language) is a shading language for WebGPU, similar to GLSL or HLSL.
    const shader = device.createShaderModule({
        code: `
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
`,
    });

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
    // There are two ways to create a pipeline layout:
    //   1. Automatic: the browser derives layouts from shader declarations. Convenient for simple pipelines.
    //   2. Explicitly create a pipeline layout with bind group layouts, and use it.
    //      Useful for sharing bind groups between pipelines and controlling compatibility.
    //      example:
    //       const pipelineLayout = device.createPipelineLayout({
    //                                 bindGroupLayouts: [layout0, layout1],
    //                               });
    //       const pipeline = device.createRenderPipeline({
    //             layout: pipelineLayout,
    //             ...
    //       });
    const pipeline = device.createRenderPipeline({
        layout: "auto",
        vertex: {
            module: shader,
            entryPoint: "vertex_main",
            buffers: [
                {
                    arrayStride: 2 * 4, // two float32 values
                    stepMode: "vertex",
                    attributes: [
                        {
                            shaderLocation: 0,
                            offset: 0,
                            format: "float32x2",
                        },
                    ],
                },
                {
                    arrayStride: 4 * 1, // four unsigned byte values: stride must be multiple of 4 bytes for alignment
                    stepMode: "vertex",
                    attributes: [
                        {
                            shaderLocation: 1,
                            offset: 0,
                            format: "unorm8x4", // normalized unsigned byte values
                        },
                    ],
                },
            ],
        },
        primitive: {
            topology: "triangle-list",
            frontFace: "ccw",
            cullMode: "none",
        },
        fragment: {
            module: shader,
            entryPoint: "fragment_main",
            targets: [{ format: textureFormat }],
        },
    });

    // 7. Event handling
    // Callbacks for keyboard and mouse events can be registered on the canvas.
    canvas.tabIndex = 0; // makes the canvas focusable so it can receive key events
    canvas.addEventListener("keydown", (event) => {
        console.log(event.key);
    });
    canvas.addEventListener("pointerdown", (event) => {
        console.log(event.offsetX, event.offsetY, event.button);
    });

    // 8. Render one frame

    // To render a frame, we need to:
    //   1. Create a command encoder
    //   2. Begin a render pass
    //   3. Set the pipeline and vertex buffers
    //   4. Draw the vertices
    //   5. End the render pass
    //   6. Submit the command buffer to the queue

    // Renders one frame with raw WebGPU calls: begins a render pass on the
    // canvas's current texture, binds the pipeline/vertex buffers, draws the
    // 3 vertices, and submits. Called once below, as this is a static scene.
    function draw() {
        const texture = context.getCurrentTexture();
        const view = texture.createView();

        // Encoder and render pass are one-use objects, and must be created for each frame.
        const encoder = device.createCommandEncoder();

        // A render pass is a sequence of rendering commands that are executed together, and can be used to render to one or more textures.
        const renderPass = encoder.beginRenderPass({
            colorAttachments: [
                {
                    view: view, // the texture view to render to
                    loadOp: "clear", // "clear" to clear the texture, "load" to keep the previous contents
                    storeOp: "store", // "store" to keep the contents after the pass, "discard" to discard the contents
                    clearValue: { r: 1.0, g: 1.0, b: 1.0, a: 1.0 }, // white background
                },
            ],
        });

        renderPass.setPipeline(pipeline);
        // set the vertex buffers: first buffer is at slot 0, second buffer is at slot 1
        renderPass.setVertexBuffer(0, coordBuffer);
        renderPass.setVertexBuffer(1, colorBuffer);
        renderPass.draw(3); // indicates the number of vertices to draw, starting from vertex 0
        renderPass.end(); // end the render pass

        // complete the command encoder
        const commandBuffer = encoder.finish();

        // a queue is used to sends work and data from the CPU to the GPU: commands as executed in order
        // each command buffer can be submitted only once
        // execution is asynchronous, and the CPU can continue to run while the GPU is working
        device.queue.submit([commandBuffer]); // submit the command buffer to the queue
    }

    // request the frame to be drawn; with a static scene like this one, a
    // single call is enough -- the browser's "on demand" equivalent of
    // canvas.request_draw(draw). There is no run loop to start (unlike
    // rendercanvas's loop.run()): the browser's own event loop is already running.
    draw();
}

main();
