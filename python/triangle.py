
from typing import Any

import wgpu
from array import array
from rendercanvas.glfw import RenderCanvas, loop


# 1. Window/canvas
# RenderCanvas(
#    size = (width, height),
#    title ="Canvas title",
#    update_mode = "ondemand" | "continuous" | "manual", ,
#       ondemand: for static scene
#           redraw only when requested (default): canvas.request_draw(), or canvas.request_redraw(), or to maintain min_fps
#       continuous: for animation
#           redraw continuously at max_fps
#       manual: for debug or custom loops
#           redraw only when requested: canvas.force_draw() or canvas.force_redraw()
#    min_fps = 0.0,
#    max_fps = 30.0,
#    vsync = True,
#    present_method = None | "screen" | "bitmap",
#)
canvas = RenderCanvas(
    size=(640, 480),
    title="WebGPU triangle",
    update_mode="ondemand",
)

# get the WebGPU context from the canvas
context: Any = canvas.get_context("wgpu")

# 2. Adapter and device
# adapter is the interface to the GPU: integrated, discrete, or virtual GPU (SW). 
# request a GPU adapter with high-performance preference and compatible with the context
adapter = wgpu.gpu.request_adapter_sync(
    power_preference="high-performance",
    canvas=context,
)

# device is the interface to the GPU, and is used to create resources and submit work
# request a device from the adapter
device = adapter.request_device_sync()

# 3. Configure canvas
# retrieve the preferred texture format for the adapter and configure the context with the device and format
texture_format = context.get_preferred_format(adapter) # e.g. "bgra8unorm", "rgba8unorm-srgb", "rgba16float"

# configure the context with the device and format
# effect of alpha_mode: canvas composting with the window background, or not
#   premultiplied: Cout = Cs + Cd x (1 - As), Aout = As + Ad x (1 - As)
#   unpremultiplied: Cout = Cs x As + Cd x (1 - As), Aout = As + Ad x (1 - As)
#   opaque: Cout = Cs, Aout = 1
# canvas composition is a final imagem composition; it takes place after blending 
context.configure(
    device=device,
    format=texture_format,
    alpha_mode="opaque",
)

# 4. Vertex data: coordinates and colors
# array is a contiguous binary data suitable for uploading to a GPU buffer
# (numpy array is also suitable)
coords = array(
    "f", # float32
    [
         0.0,  0.7,
        -0.7, -0.7,
         0.7, -0.7,
    ],
)
colors = array(
    "B", # unsigned char
    [
        255, 0, 0, 255,
        0, 255, 0, 255,
        0, 0, 255, 255,
    ],
)

# GPU buffers are used to store data on the GPU, and can be used as vertex buffers, index buffers, uniform buffers, storage buffers, etc.
# create a GPU buffer for the vertex coordinates and colors
#   possible usage flags: 
#    wgpu.BufferUsage.VERTEX: for vertex buffers
#    wgpu.BufferUsage.INDEX: for index buffers
#    wgpu.BufferUsage.UNIFORM: for uniform buffers
#    wgpu.BufferUsage.STORAGE: for storage buffers
#   that can be combined with:
#    wgpu.BufferUsage.COPY_SRC: for copying data from the buffer
#    wgpu.BufferUsage.COPY_DST: for copying data to the buffer
coord_buffer = device.create_buffer_with_data(
    data=coords,
    usage=wgpu.BufferUsage.VERTEX,
)
color_buffer = device.create_buffer_with_data(
    data=colors,
    usage=wgpu.BufferUsage.VERTEX,
)


# 5. Shaders
# Shaders are programs that run on the GPU, and are used to process vertex data and fragment data.
# WGSL (WebGPU Shading Language) is a shading language for WebGPU, similar to GLSL or HLSL.
shader = device.create_shader_module(
    code="""
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
"""
)


# 6. Pipeline (immutable)

# A pipeline is a configured GPU program describing how a draw or compute operation works.
# A render pipeline contains:
#   Vertex and fragment shaders
#   Vertex-buffer layouts
#   Primitive type: triangles, lines, points
#   Rasterization and face-culling settings
#   Color formats and blending
#   Depth/stencil and multisampling settings
#   Resource-binding layout

# Pipeline layout describes the resources (buffers, textures, samplers) that are used by the shaders.
# There are two ways to create a pipeline layout:
#   1. Automatic: wgpu derives layouts from shader declarations. Convenient for simple pipelines.
#   2. Explicitly create a pipeline layout with bind group layouts, and use it. 
#      Useful for sharing bind groups between pipelines and controlling compatibility.
#      example:
#       pipeline_layout = device.create_pipeline_layout(
#                           bind_group_layouts=[layout0, layout1]
#                         )
#       pipeline = device.create_render_pipeline(
#             layout=pipeline_layout,
#             ...
#       )
pipeline = device.create_render_pipeline(
    layout="auto",
    vertex={
        "module": shader,
        "entry_point": "vertex_main",
        "buffers": [
            {
                "array_stride": 2 * 4,  # two float32 values
                "step_mode": "vertex",
                "attributes": [
                    {
                        "shader_location": 0,
                        "offset": 0,
                        "format": "float32x2",
                    }
                ]
            },
            {
                "array_stride": 4 * 1,  # four unsigned byte values: stride must be multiple of 4 bytes for alignment
                "step_mode": "vertex",
                "attributes": [
                    {
                        "shader_location": 1,
                        "offset": 0,
                        "format": "unorm8x4",  # normalized unsigned byte values
                    } 
                ]
            }
        ],
    },
    primitive={
        "topology": "triangle-list",
        "front_face": "ccw",
        "cull_mode": "none",
    },
    fragment={
        "module": shader,
        "entry_point": "fragment_main",
        "targets": [{"format": texture_format}],
    },
)

# 7. Event handling
# Callbacks for keyboard and mouse events can be registered with the canvas.
def on_key(event):
    print(event["key"])
canvas.add_event_handler(on_key, "key_down")

def on_pointer(event):
    print(event["x"], event["y"], event["button"])
canvas.add_event_handler(on_pointer, "pointer_down")

# 8. Render one frame

# To render a frame, we need to:
#   1. Create a command encoder
#   2. Begin a render pass
#   3. Set the pipeline and vertex buffers
#   4. Draw the vertices
#   5. End the render pass
#   6. Submit the command buffer to the queue


def draw():
    """Renders one frame with raw wgpu calls: begins a render pass on the
    canvas's current texture, binds the pipeline/vertex buffers, draws the
    3 vertices, and submits. Registered as the canvas's redraw callback
    below."""
    texture = context.get_current_texture()
    view = texture.create_view()

# Encoder and render pass are one-use objects, and must be created for each frame.
    encoder = device.create_command_encoder()

# A render pass is a sequence of rendering commands that are executed together, and can be used to render to one or more textures.
    render_pass = encoder.begin_render_pass(
        color_attachments=[
            {
                "view": view, # the texture view to render to
                "load_op": "clear",  # "clear" to clear the texture, "load" to keep the previous contents
                "store_op": "store", # "store" to keep the contents after the pass, "discard" to discard the contents
                "clear_value": (1.0, 1.0, 1.0, 1.0),  # white background
            }
        ],
        depth_stencil_attachment = None
    )

    render_pass.set_pipeline(pipeline)
    # set the vertex buffers: first buffer is at slot 0, second buffer is at slot 1
    render_pass.set_vertex_buffer(0, coord_buffer)
    render_pass.set_vertex_buffer(1, color_buffer)
    render_pass.draw(3) # indicates the number of vertices to draw, starting from vertex 0
    render_pass.end() # end the render pass

    # complete the command encoder
    command_buffer = encoder.finish()

    # a queue is used to sends work and data from the CPU to the GPU: commands as executed in order
    # each command buffer can be submitted only once
    # executuion is assynchronous, and the CPU can continue to run while the GPU is working
    device.queue.submit([command_buffer]) # submit the command buffer to the queue

# set the draw function to be called when the canvas needs to be redrawn
# with "continuous" update mode, the draw function is called continuously at max_fps
# with "ondemand" update mode, the draw function is called only when requested
canvas.request_draw(draw)

# start the GLFW event loop: process events and render frames
loop.run()


