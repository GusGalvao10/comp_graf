from array import array
from dataclasses import dataclass, field
import math
from pathlib import Path
import time
from typing import Any, Optional
from PIL import Image
import wgpu
from rendercanvas.glfw import RenderCanvas, loop


canvas = RenderCanvas(
    size=(800, 600),
    title="sistema solar P1 comp graf",
    update_mode="continuous",
)
context: Any = canvas.get_context("wgpu")

adapter = wgpu.gpu.request_adapter_sync(
    power_preference="high-performance",
    canvas=canvas,
)
device = adapter.request_device_sync()
texture_format = context.get_preferred_format(adapter)
context.configure(device=device, format=texture_format, alpha_mode="opaque")
DIRETORIO = Path(__file__).with_name("texturas")


def identity_matrix() -> array:
    return array(
        "f",
        [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ],
    )


def translate_scale(x: float, y: float, scale: float) -> array:
    """Matriz column-major: T(x, y) @ S(scale)"""
    return array(
        "f",
        [
            scale, 0.0, 0.0, 0.0,
            0.0, scale, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            x, y, 0.0, 1.0,
        ],
    )


def scale_matrix(x_scale: float, y_scale: float) -> array:
    """Matriz column-major de escala não uniforme"""
    return array(
        "f",
        [
            x_scale, 0.0, 0.0, 0.0,
            0.0, y_scale, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ],
    )


def rotation_matrix(angle: float) -> array:
    """Cria uma rotação 2D ao redor da origem, em radianos."""
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return array(
        "f",
        [
            cosine, sine, 0.0, 0.0,
            -sine, cosine, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ],
    )


def multiply_matrices(left: array, right: array) -> array:
    """Multiplica matrizes 4x4 armazenadas em column-major."""
    result = array("f", [0.0] * 16)
    for column in range(4):
        for row in range(4):
            result[column * 4 + row] = sum(
                left[index * 4 + row] * right[column * 4 + index]
                for index in range(4)
            )
    return result


def create_disk_tex(segments: int = 64) -> tuple[array, array, array]:
    """Cria um disco e suas coordenadas de textura em buffers separados."""
    vertices = array("f", [0.0, 0.0])
    texcoords = array("f", [0.5, 0.5])
    for index in range(segments + 1):
        angle = 2.0 * math.pi * index / segments
        x, y = math.cos(angle), math.sin(angle)
        vertices.extend((x, y))
        texcoords.extend((0.5 + 0.5 * x, 0.5 - 0.5 * y))

    indices = array("I")
    for index in range(segments):
        indices.extend((0, index + 1, index + 2))
    return vertices, texcoords, indices


def create_quad_tex() -> tuple[array, array, array]:
    """Cria um quad com coordenadas de textura para preencher a tela."""
    vertices = array("f", [-1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0])
    texcoords = array("f", [0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    indices = array("I", [0, 1, 2, 0, 2, 3])
    return vertices, texcoords, indices


disk_vertices, disk_texcoords, disk_indices = create_disk_tex()
disk_vertex_buffer = device.create_buffer_with_data(
    data=disk_vertices,
    usage=wgpu.BufferUsage.VERTEX,
)
disk_texcoord_buffer = device.create_buffer_with_data(
    data=disk_texcoords,
    usage=wgpu.BufferUsage.VERTEX,
)
disk_index_buffer = device.create_buffer_with_data(
    data=disk_indices,
    usage=wgpu.BufferUsage.INDEX,
)
quad_vertices, quad_texcoords, quad_indices = create_quad_tex()
quad_vertex_buffer = device.create_buffer_with_data(
    data=quad_vertices,
    usage=wgpu.BufferUsage.VERTEX,
)
quad_texcoord_buffer = device.create_buffer_with_data(
    data=quad_texcoords,
    usage=wgpu.BufferUsage.VERTEX,
)
quad_index_buffer = device.create_buffer_with_data(
    data=quad_indices,
    usage=wgpu.BufferUsage.INDEX,
)

aspect_buffer = device.create_buffer_with_data(
    data=array("f", [1.0]),
    usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
)

shader = device.create_shader_module(
    code="""
struct ObjectData {
    model: mat4x4<f32>,
    color: vec4<f32>,
};

@group(0) @binding(0) var<uniform> aspect_ratio: f32;
@group(1) @binding(0) var<uniform> object: ObjectData;
@group(2) @binding(0) var astro_tex: texture_2d<f32>;
@group(2) @binding(1) var astro_smp: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) texcoord: vec2<f32>,
};

@vertex
fn vertex_main(
    @location(0) position: vec2<f32>,
    @location(1) texcoord: vec2<f32>,
) -> VertexOutput {
    var output: VertexOutput;
    let world_position = object.model * vec4<f32>(position, 0.0, 1.0);
    output.position = vec4<f32>(
        world_position.x * aspect_ratio,
        world_position.y,
        world_position.z,
        world_position.w,
    );
    output.color = object.color;
    output.texcoord = texcoord;
    return output;
}

@fragment
fn fragment_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let texel = textureSample(astro_tex, astro_smp, input.texcoord);
    return input.color * texel;
}
"""
)

pipeline = device.create_render_pipeline(
    layout="auto",
    vertex={
        "module": shader,
        "entry_point": "vertex_main",
        "buffers": [
            {
                "array_stride": 2 * 4,
                "step_mode": "vertex",
                "attributes": [
                    {"shader_location": 0, "offset": 0, "format": "float32x2"}
                ],
            },
            {
                "array_stride": 2 * 4,
                "step_mode": "vertex",
                "attributes": [
                    {"shader_location": 1, "offset": 0, "format": "float32x2"}
                ],
            },
        ],
    },
    primitive={"topology": "triangle-list", "front_face": "ccw", "cull_mode": "none"},
    fragment={
        "module": shader,
        "entry_point": "fragment_main",
        "targets": [{"format": texture_format}],
    },
)

scene_bind_group = device.create_bind_group(
    layout=pipeline.get_bind_group_layout(0),
    entries=[{"binding": 0, "resource": {"buffer": aspect_buffer, "size": 4}}],
)

astro_sampler = device.create_sampler(
    address_mode_u="repeat",
    address_mode_v="repeat",
    mag_filter="linear",
    min_filter="linear",
    mipmap_filter="linear",
)


def create_texture_bind_group(image_path: Path) -> Any:
    """Carrega uma imagem RGBA e cria seus recursos de textura na GPU."""
    image = Image.open(image_path).convert("RGBA")
    width, height = image.size
    texture = device.create_texture(
        size=(width, height, 1),
        format="rgba8unorm-srgb",
        usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
    )
    device.queue.write_texture(
        {"texture": texture},
        image.tobytes(),
        {"bytes_per_row": width * 4, "rows_per_image": height},
        (width, height, 1),
    )
    texture_view = texture.create_view()
    return device.create_bind_group(
        layout=pipeline.get_bind_group_layout(2),
        entries=[
            {"binding": 0, "resource": texture_view},
            {"binding": 1, "resource": astro_sampler},
        ],
    )


@dataclass
class Disk:
    """Geometria compartilhada pelos três astros."""

    def draw(self, render_pass) -> None:
        render_pass.set_vertex_buffer(0, disk_vertex_buffer)
        render_pass.set_vertex_buffer(1, disk_texcoord_buffer)
        render_pass.set_index_buffer(disk_index_buffer, "uint32")
        render_pass.draw_indexed(len(disk_indices))


@dataclass
class Quad:
    """Geometria do retângulo que recebe a textura de fundo."""

    def draw(self, render_pass) -> None:
        render_pass.set_vertex_buffer(0, quad_vertex_buffer)
        render_pass.set_vertex_buffer(1, quad_texcoord_buffer)
        render_pass.set_index_buffer(quad_index_buffer, "uint32")
        render_pass.draw_indexed(len(quad_indices))


@dataclass
class Node:
    """Nó mínimo de grafo de cena: transformação herdada e forma local."""

    name: str
    local_matrix: array = field(default_factory=identity_matrix)
    color: Optional[tuple[float, float, float, float]] = None
    shape: Optional[Any] = None
    texture_path: Optional[Path] = None
    children: list["Node"] = field(default_factory=list)
    object_buffer: Any = field(init=False, default=None)
    object_bind_group: Any = field(init=False, default=None)
    texture_bind_group: Any = field(init=False, default=None)

    def __post_init__(self) -> None:
        if self.color is not None:
            data = array("f", self.local_matrix)
            data.extend(self.color)
            self.object_buffer = device.create_buffer_with_data(
                data=data,
                usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
            )
            self.object_bind_group = device.create_bind_group(
                layout=pipeline.get_bind_group_layout(1),
                entries=[
                    {
                        "binding": 0,
                        "resource": {"buffer": self.object_buffer, "size": 80},
                    }
                ],
            )
        if self.shape is not None and self.texture_path is not None:
            self.texture_bind_group = create_texture_bind_group(self.texture_path)

    def add_child(self, child: "Node") -> None:
        self.children.append(child)

    def prepare(self, parent_matrix: array) -> None:
        """Calcula matrizes e transfere dados antes de abrir a render pass."""
        world_matrix = multiply_matrices(parent_matrix, self.local_matrix)
        if self.shape is not None and self.object_buffer is not None:
            data = array("f", world_matrix)
            data.extend(self.color)
            device.queue.write_buffer(self.object_buffer, 0, data)

        for child in self.children:
            child.prepare(world_matrix)

    def render(self, render_pass) -> None:
        if self.shape is not None and self.object_bind_group is not None:
            render_pass.set_bind_group(1, self.object_bind_group)
            render_pass.set_bind_group(2, self.texture_bind_group)
            self.shape.draw(render_pass)

        for child in self.children:
            child.render(render_pass)


# A posição da Terra herda sua órbita. A rotação própria da Terra e a órbita
# da Lua são ramos distintos para a Lua não herdar a rotação terrestre.
disk = Disk()
quad = Quad()
root = Node("raiz")
background = Node(
    "fundo",
    color=(1.0, 1.0, 1.0, 1.0),
    shape=quad,
    texture_path=DIRETORIO / "universo.jpg",
)
sun = Node(
    "sol",
    local_matrix=translate_scale(0.0, 0.0, 0.22),
    color=(1.0, 1.0, 1.0, 1.0),
    shape=disk,
    texture_path=DIRETORIO / "sol.png",
)
mercury_orbit = Node("órbita de mercúrio")
mercury = Node(
    "mercúrio",
    local_matrix=translate_scale(0.38, 0.0, 0.055),
    color=(1.0, 1.0, 1.0, 1.0),
    shape=disk,
    texture_path=DIRETORIO / "mercurio.png",
)
earth_orbit = Node("órbita da terra")
earth = Node(
    "posição da terra",
    local_matrix=translate_scale(0.72, 0.0, 1.0),
)
earth_rotation = Node("rotação da terra")
earth_body = Node(
    "terra",
    local_matrix=translate_scale(0.0, 0.0, 0.09),
    color=(1.0, 1.0, 1.0, 1.0),
    shape=disk,
    texture_path=DIRETORIO / "terra.png",
)
moon_orbit = Node("órbita da lua")
moon = Node(
    "lua",
    local_matrix=translate_scale(0.18, 0.0, 0.045),
    color=(1.0, 1.0, 1.0, 1.0),
    shape=disk,
    texture_path=DIRETORIO / "Lua.png",
)
root.add_child(background)
root.add_child(sun)
root.add_child(mercury_orbit)
root.add_child(earth_orbit)
mercury_orbit.add_child(mercury)
earth_orbit.add_child(earth)
earth.add_child(earth_rotation)
earth.add_child(moon_orbit)
earth_rotation.add_child(earth_body)
moon_orbit.add_child(moon)


@dataclass
class OrbitEngine:
    """Atualiza a rotação de um nó de órbita a partir do tempo decorrido."""

    orbit_node: Node
    period_seconds: float = 12.0
    angle: float = 0.0

    def update(self, dt: float) -> None:
        self.angle = (self.angle + 2.0 * math.pi * dt / self.period_seconds) % math.tau
        self.orbit_node.local_matrix = rotation_matrix(self.angle)


earth_engine = OrbitEngine(earth_orbit, period_seconds=12.0)
earth_rotation_engine = OrbitEngine(earth_rotation, period_seconds=2.0)
moon_engine = OrbitEngine(moon_orbit, period_seconds=3.0)
mercury_engine = OrbitEngine(mercury_orbit, period_seconds=7.0)
last_time = time.perf_counter()


def draw() -> None:
    global last_time
    current_time = time.perf_counter()
    dt = current_time - last_time
    earth_engine.update(dt)
    earth_rotation_engine.update(dt)
    moon_engine.update(dt)
    mercury_engine.update(dt)
    last_time = current_time

    width, height = canvas.get_physical_size()
    if width > 0 and height > 0:
        device.queue.write_buffer(aspect_buffer, 0, array("f", [height / width]))
        background.local_matrix = scale_matrix(width / height, 1.0)
    root.prepare(identity_matrix())

    texture = context.get_current_texture()
    view = texture.create_view()
    encoder = device.create_command_encoder()
    render_pass = encoder.begin_render_pass(
        color_attachments=[
            {
                "view": view,
                "load_op": "clear",
                "store_op": "store",
                "clear_value": (0.015, 0.025, 0.070, 1.0),
            }
        ],
        depth_stencil_attachment=None,
    )
    render_pass.set_pipeline(pipeline)
    render_pass.set_bind_group(0, scene_bind_group)
    root.render(render_pass)
    render_pass.end()
    device.queue.submit([encoder.finish()])


canvas.request_draw(draw)
loop.run()
