#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <GLFW/glfw3.h>
#include <glfw3webgpu.h>
#include <webgpu/webgpu.h>

namespace {

constexpr float kPi = 3.14159265358979323846f;

struct Vec3 { float x, y, z; };

Vec3 operator+(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
Vec3 operator-(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
Vec3 operator*(Vec3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }
float dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
Vec3 cross(Vec3 a, Vec3 b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
float length(Vec3 a) { return std::sqrt(dot(a, a)); }
Vec3 normalize(Vec3 a) { float l = length(a); return {a.x / l, a.y / l, a.z / l}; }

using Mat4 = std::array<float, 16>; // column-major

Mat4 mat4_identity() {
    Mat4 m = {};
    m[0] = m[5] = m[10] = m[15] = 1.0f;
    return m;
}

Mat4 mat4_mul(const Mat4 &a, const Mat4 &b) {
    Mat4 c = {};
    for (int col = 0; col < 4; ++col)
        for (int row = 0; row < 4; ++row) {
            float s = 0.0f;
            for (int k = 0; k < 4; ++k) s += a[k * 4 + row] * b[col * 4 + k];
            c[col * 4 + row] = s;
        }
    return c;
}

Mat4 mat4_translate(Vec3 t) {
    Mat4 m = mat4_identity();
    m[12] = t.x; m[13] = t.y; m[14] = t.z;
    return m;
}

Mat4 mat4_scale(Vec3 s) {
    Mat4 m = mat4_identity();
    m[0] = s.x; m[5] = s.y; m[10] = s.z;
    return m;
}

Mat4 translate_scale(Vec3 position, Vec3 scale) {
    return mat4_mul(mat4_translate(position), mat4_scale(scale));
}

Mat4 mat4_rotate(float angle, Vec3 axis) {
    Vec3 a = normalize(axis);
    float c = std::cos(angle), s = std::sin(angle), t = 1.0f - c;
    Mat4 m = mat4_identity();
    m[0] = t * a.x * a.x + c;       m[1] = t * a.x * a.y + s * a.z; m[2]  = t * a.x * a.z - s * a.y;
    m[4] = t * a.x * a.y - s * a.z; m[5] = t * a.y * a.y + c;       m[6]  = t * a.y * a.z + s * a.x;
    m[8] = t * a.x * a.z + s * a.y; m[9] = t * a.y * a.z - s * a.x; m[10] = t * a.z * a.z + c;
    return m;
}

Mat4 mat4_look_at(Vec3 eye, Vec3 center, Vec3 up) {
    Vec3 f = normalize(center - eye);
    Vec3 s = normalize(cross(f, up));
    Vec3 u = cross(s, f);
    Mat4 m = mat4_identity();
    m[0] = s.x;  m[4] = s.y;  m[8]  = s.z;
    m[1] = u.x;  m[5] = u.y;  m[9]  = u.z;
    m[2] = -f.x; m[6] = -f.y; m[10] = -f.z;
    m[12] = -dot(s, eye);
    m[13] = -dot(u, eye);
    m[14] = dot(f, eye);
    return m;
}

// profundidade em [0, 1], como a WebGPU espera
Mat4 mat4_perspective(float fovy, float aspect, float znear, float zfar) {
    float f = 1.0f / std::tan(fovy * 0.5f);
    Mat4 m = {};
    m[0] = f / aspect;
    m[5] = f;
    m[10] = zfar / (znear - zfar);
    m[11] = -1.0f;
    m[14] = (znear * zfar) / (znear - zfar);
    return m;
}

// inversa-transposta do bloco 3x3: colunas (a1xa2, a2xa0, a0xa1) / det
Mat4 mat4_normal_matrix(const Mat4 &mv) {
    Vec3 a0 = {mv[0], mv[1], mv[2]};
    Vec3 a1 = {mv[4], mv[5], mv[6]};
    Vec3 a2 = {mv[8], mv[9], mv[10]};
    float det = dot(a0, cross(a1, a2));
    float inv = (std::fabs(det) > 1e-12f) ? 1.0f / det : 0.0f;
    Vec3 c0 = cross(a1, a2) * inv;
    Vec3 c1 = cross(a2, a0) * inv;
    Vec3 c2 = cross(a0, a1) * inv;
    Mat4 m = mat4_identity();
    m[0] = c0.x; m[1] = c0.y; m[2]  = c0.z;
    m[4] = c1.x; m[5] = c1.y; m[6]  = c1.z;
    m[8] = c2.x; m[9] = c2.y; m[10] = c2.z;
    return m;
}

Vec3 transform_point(const Mat4 &m, Vec3 p) {
    return {m[0] * p.x + m[4] * p.y + m[8]  * p.z + m[12],
            m[1] * p.x + m[5] * p.y + m[9]  * p.z + m[13],
            m[2] * p.x + m[6] * p.y + m[10] * p.z + m[14]};
}

Vec3 transform_direction(const Mat4 &m, Vec3 d) {
    return {m[0] * d.x + m[4] * d.y + m[8]  * d.z,
            m[1] * d.x + m[5] * d.y + m[9]  * d.z,
            m[2] * d.x + m[6] * d.y + m[10] * d.z};
}

void set4(float *dst, Vec3 v, float w) { dst[0] = v.x; dst[1] = v.y; dst[2] = v.z; dst[3] = w; }

struct LightUniforms {
    float position[4];
    float direction[4];
    float spot[4];       // x = cos(theta), y = expoente, z = é spot, w = ligada
    float ambient[4];
    float diffuse[4];
    float specular[4];
};
static_assert(sizeof(LightUniforms) == 96, "LightUniforms");

struct GlobalUniforms {
    float projection[16];
    float clip_plane[4];
    float fog_color[4];
    float fog_params[4]; // x = início, y = fim, z = fog ligada, w = corte ligado
    LightUniforms lights[2];
};
static_assert(sizeof(GlobalUniforms) == 304, "GlobalUniforms");

struct ObjectUniforms {
    float modelview[16];
    float normal_matrix[16];
    float base_color[4];
    float specular_color[4]; // w = brilho
    float uv_scale[4];
};
static_assert(sizeof(ObjectUniforms) == 176, "ObjectUniforms");

enum class LightSpace { World, Camera };

class Light {
public:
    LightSpace space = LightSpace::World;
    Vec3 position = {0.0f, 0.0f, 0.0f};
    Vec3 ambient = {0.0f, 0.0f, 0.0f};
    Vec3 diffuse = {1.0f, 1.0f, 1.0f};
    Vec3 specular = {1.0f, 1.0f, 1.0f};
    bool enabled = true;

    virtual ~Light() = default;

    Mat4 get_mapping_matrix(const Mat4 &view) const {
        return space == LightSpace::Camera ? mat4_identity() : view;
    }

    void write_fields(LightUniforms &u, const Mat4 &view) const {
        std::memset(&u, 0, sizeof(u));
        Mat4 m = get_mapping_matrix(view);
        set4(u.position, transform_point(m, position), 1.0f);
        set4(u.ambient, ambient, 1.0f);
        set4(u.diffuse, diffuse, 1.0f);
        set4(u.specular, specular, 1.0f);
        write_shape(u, m);
        u.spot[3] = enabled ? 1.0f : 0.0f;
    }

protected:
    virtual void write_shape(LightUniforms &u, const Mat4 &m) const = 0;
};

class PointLight : public Light {
protected:
    void write_shape(LightUniforms &u, const Mat4 &) const override {
        u.spot[0] = -1.0f;
    }
};

class SpotLight : public Light {
public:
    Vec3 direction = {0.0f, -1.0f, 0.0f};
    float angle = 0.4f;
    float exponent = 0.0f;

protected:
    void write_shape(LightUniforms &u, const Mat4 &m) const override {
        set4(u.direction, normalize(transform_direction(m, direction)), 0.0f);
        u.spot[0] = std::cos(angle);
        u.spot[1] = exponent;
        u.spot[2] = 1.0f;
    }
};

Vec3 arcball_map(double width, double height, double x, double y) {
    double r = (width < height ? width : height) / 2.0;
    double X = (x - width / 2.0) / r;
    double Y = (y - height / 2.0) / r;
    double l = std::sqrt(X * X + Y * Y);
    double Z = 0.0;
    if (l <= 1.0) {
        Z = std::sqrt(1.0 - l * l);
    } else {
        X /= l;
        Y /= l;
    }
    return {static_cast<float>(X), static_cast<float>(Y), static_cast<float>(Z)};
}

struct Arcball {
    float distance;
    double x0 = 0.0, y0 = 0.0;
    Mat4 mat = mat4_identity();
    bool dragging = false;

    explicit Arcball(float d) : distance(d) {}

    void init_mouse_motion(double x, double y) { x0 = x; y0 = y; }

    void accumulate_mouse_motion(double x, double y, double width, double height) {
        if (x == x0 && y == y0) return;
        Vec3 u = arcball_map(width, height, x0, y0);
        Vec3 v = arcball_map(width, height, x, y);
        x0 = x; y0 = y;
        Vec3 a = cross(u, v);
        float cross_len = length(a);
        if (cross_len < 1e-9f) return;
        float theta = 2.0f * std::atan2(cross_len, dot(u, v));
        Mat4 m = mat4_translate({0.0f, 0.0f, -distance});
        m = mat4_mul(m, mat4_rotate(theta, a * (1.0f / cross_len)));
        m = mat4_mul(m, mat4_translate({0.0f, 0.0f, distance}));
        mat = mat4_mul(m, mat);
    }
};

struct App {
    Arcball arcball;
    bool fog = true;
    bool clip = false;
    bool light2 = true;
    explicit App(float d) : arcball(d) {}
};

// o GLFW usa coordenadas lógicas da janela, por isso glfwGetWindowSize
void mouse_button_callback(GLFWwindow *window, int button, int action, int) {
    if (button != GLFW_MOUSE_BUTTON_LEFT) return;
    App *app = static_cast<App *>(glfwGetWindowUserPointer(window));
    if (!app) return;
    if (action == GLFW_PRESS) {
        double x, y;
        int w, h;
        glfwGetCursorPos(window, &x, &y);
        glfwGetWindowSize(window, &w, &h);
        app->arcball.dragging = true;
        app->arcball.init_mouse_motion(x, h - y);
    } else if (action == GLFW_RELEASE) {
        app->arcball.dragging = false;
    }
}

void cursor_pos_callback(GLFWwindow *window, double x, double y) {
    App *app = static_cast<App *>(glfwGetWindowUserPointer(window));
    if (!app || !app->arcball.dragging) return;
    int w, h;
    glfwGetWindowSize(window, &w, &h);
    if (w <= 0 || h <= 0) return;
    app->arcball.accumulate_mouse_motion(x, h - y, w, h);
}

void key_callback(GLFWwindow *window, int key, int, int action, int) {
    if (action != GLFW_PRESS) return;
    App *app = static_cast<App *>(glfwGetWindowUserPointer(window));
    if (!app) return;
    if (key == GLFW_KEY_ESCAPE) {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    } else if (key == GLFW_KEY_F) {
        app->fog = !app->fog;
        std::printf("Neblina: %s\n", app->fog ? "ligada" : "desligada");
    } else if (key == GLFW_KEY_C) {
        app->clip = !app->clip;
        std::printf("Plano de corte: %s\n", app->clip ? "ligado" : "desligado");
    } else if (key == GLFW_KEY_L) {
        app->light2 = !app->light2;
        std::printf("Segunda luz: %s\n", app->light2 ? "ligada" : "desligada");
    }
}

struct MeshData {
    std::vector<float> vertices; // x y z nx ny nz u v
    std::vector<uint32_t> indices;
};

void push_vertex(MeshData &m, Vec3 p, Vec3 n, float u, float v) {
    const float data[8] = {p.x, p.y, p.z, n.x, n.y, n.z, u, v};
    m.vertices.insert(m.vertices.end(), data, data + 8);
}

MeshData make_cube() {
    MeshData m;
    // normal, u, v (com u x v = normal)
    const Vec3 faces[6][3] = {
        {{ 1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
        {{-1, 0, 0}, {0, 0, 1}, {0, 1, 0}},
        {{ 0, 1, 0}, {0, 0, 1}, {1, 0, 0}},
        {{ 0,-1, 0}, {1, 0, 0}, {0, 0, 1}},
        {{ 0, 0, 1}, {1, 0, 0}, {0, 1, 0}},
        {{ 0, 0,-1}, {0, 1, 0}, {1, 0, 0}},
    };
    const float uvs[4][2] = {{0, 1}, {1, 1}, {1, 0}, {0, 0}};
    for (int f = 0; f < 6; ++f) {
        Vec3 n = faces[f][0], u = faces[f][1], v = faces[f][2];
        uint32_t base = static_cast<uint32_t>(m.vertices.size() / 8);
        Vec3 c = n * 0.5f;
        push_vertex(m, c - u * 0.5f - v * 0.5f, n, uvs[0][0], uvs[0][1]);
        push_vertex(m, c + u * 0.5f - v * 0.5f, n, uvs[1][0], uvs[1][1]);
        push_vertex(m, c + u * 0.5f + v * 0.5f, n, uvs[2][0], uvs[2][1]);
        push_vertex(m, c - u * 0.5f + v * 0.5f, n, uvs[3][0], uvs[3][1]);
        m.indices.insert(m.indices.end(), {base, base + 1, base + 2, base, base + 2, base + 3});
    }
    return m;
}

MeshData make_sphere(int stacks, int slices) {
    MeshData m;
    for (int i = 0; i <= stacks; ++i) {
        float theta = kPi * static_cast<float>(i) / stacks;
        for (int j = 0; j <= slices; ++j) {
            float phi = 2.0f * kPi * static_cast<float>(j) / slices;
            Vec3 p = {std::sin(theta) * std::cos(phi), std::cos(theta), std::sin(theta) * std::sin(phi)};
            push_vertex(m, p, p, static_cast<float>(j) / slices, static_cast<float>(i) / stacks);
        }
    }
    for (int i = 0; i < stacks; ++i) {
        for (int j = 0; j < slices; ++j) {
            uint32_t a = static_cast<uint32_t>(i * (slices + 1) + j);
            uint32_t b = a + static_cast<uint32_t>(slices + 1);
            m.indices.insert(m.indices.end(), {a, a + 1, b, a + 1, b + 1, b});
        }
    }
    return m;
}

struct Image {
    uint32_t width = 0, height = 0;
    std::vector<uint8_t> rgba;

    Image(uint32_t w, uint32_t h) : width(w), height(h), rgba(static_cast<size_t>(w) * h * 4, 255) {}

    void set(uint32_t x, uint32_t y, float r, float g, float b) {
        size_t i = (static_cast<size_t>(y) * width + x) * 4;
        rgba[i] = static_cast<uint8_t>(r);
        rgba[i + 1] = static_cast<uint8_t>(g);
        rgba[i + 2] = static_cast<uint8_t>(b);
    }
};

float lerpf(float a, float b, float t) { return a + (b - a) * t; }

Image make_wood(uint32_t size) {
    Image img(size, size);
    for (uint32_t y = 0; y < size; ++y)
        for (uint32_t x = 0; x < size; ++x) {
            float u = static_cast<float>(x) / size, v = static_cast<float>(y) / size;
            float rings = 0.5f + 0.5f * std::sin(2.0f * kPi * (u * 8.0f + 0.15f * std::sin(2.0f * kPi * v * 3.0f)));
            float t = std::fmin(1.0f, std::fmax(0.0f, rings + 0.04f * std::sin(2.0f * kPi * v * 40.0f)));
            img.set(x, y, lerpf(110, 175, t), lerpf(70, 120, t), lerpf(35, 70, t));
        }
    return img;
}

Image make_checker(uint32_t size, uint32_t squares) {
    Image img(size, size);
    for (uint32_t y = 0; y < size; ++y)
        for (uint32_t x = 0; x < size; ++x) {
            float c = ((x * squares / size) + (y * squares / size)) % 2 == 0 ? 255.0f : 150.0f;
            img.set(x, y, c, c, c);
        }
    return img;
}

Image make_stripes(uint32_t size, uint32_t bands) {
    Image img(size, size);
    for (uint32_t y = 0; y < size; ++y)
        for (uint32_t x = 0; x < size; ++x) {
            float c = (x * bands / size) % 2 == 0 ? 255.0f : 110.0f;
            img.set(x, y, c, c, c);
        }
    return img;
}

Image make_white() { return Image(1, 1); }

struct AdapterRequest { WGPUAdapter adapter = nullptr; };
struct DeviceRequest { WGPUDevice device = nullptr; };

void handle_request_adapter(WGPURequestAdapterStatus status, WGPUAdapter adapter,
                            WGPUStringView, void *userdata1, void *) {
    if (status == WGPURequestAdapterStatus_Success) {
        static_cast<AdapterRequest *>(userdata1)->adapter = adapter;
    }
}

void handle_request_device(WGPURequestDeviceStatus status, WGPUDevice device,
                           WGPUStringView, void *userdata1, void *) {
    if (status == WGPURequestDeviceStatus_Success) {
        static_cast<DeviceRequest *>(userdata1)->device = device;
    }
}

struct Gpu {
    WGPUDevice device = nullptr;
    WGPUQueue queue = nullptr;
    WGPUBindGroupLayout object_layout = nullptr;
    WGPUBindGroupLayout decal_layout = nullptr;
    WGPUSampler sampler = nullptr;
};
Gpu gpu;

constexpr WGPUTextureFormat kDepthFormat = WGPUTextureFormat_Depth32Float;

WGPUBuffer create_buffer_with_data(const void *data, size_t size, WGPUBufferUsage usage) {
    WGPUBufferDescriptor descriptor = {};
    descriptor.usage = usage;
    descriptor.size = size;
    descriptor.mappedAtCreation = true;
    WGPUBuffer buffer = wgpuDeviceCreateBuffer(gpu.device, &descriptor);
    std::memcpy(wgpuBufferGetMappedRange(buffer, 0, size), data, size);
    wgpuBufferUnmap(buffer);
    return buffer;
}

WGPUBuffer create_uniform_buffer(size_t size) {
    WGPUBufferDescriptor descriptor = {};
    descriptor.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
    descriptor.size = size;
    return wgpuDeviceCreateBuffer(gpu.device, &descriptor);
}

WGPUBindGroupLayout create_uniform_layout(uint64_t min_size) {
    WGPUBindGroupLayoutEntry entry = {};
    entry.binding = 0;
    entry.visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment;
    entry.buffer.type = WGPUBufferBindingType_Uniform;
    entry.buffer.minBindingSize = min_size;
    WGPUBindGroupLayoutDescriptor descriptor = {};
    descriptor.entryCount = 1;
    descriptor.entries = &entry;
    return wgpuDeviceCreateBindGroupLayout(gpu.device, &descriptor);
}

WGPUBindGroup create_uniform_bind_group(WGPUBindGroupLayout layout, WGPUBuffer buffer, uint64_t size) {
    WGPUBindGroupEntry entry = {};
    entry.binding = 0;
    entry.buffer = buffer;
    entry.size = size;
    WGPUBindGroupDescriptor descriptor = {};
    descriptor.layout = layout;
    descriptor.entryCount = 1;
    descriptor.entries = &entry;
    return wgpuDeviceCreateBindGroup(gpu.device, &descriptor);
}

WGPUBindGroupLayout create_decal_layout() {
    WGPUBindGroupLayoutEntry entries[2] = {};
    entries[0].binding = 0;
    entries[0].visibility = WGPUShaderStage_Fragment;
    entries[0].texture.sampleType = WGPUTextureSampleType_Float;
    entries[0].texture.viewDimension = WGPUTextureViewDimension_2D;
    entries[1].binding = 1;
    entries[1].visibility = WGPUShaderStage_Fragment;
    entries[1].sampler.type = WGPUSamplerBindingType_Filtering;
    WGPUBindGroupLayoutDescriptor descriptor = {};
    descriptor.entryCount = 2;
    descriptor.entries = entries;
    return wgpuDeviceCreateBindGroupLayout(gpu.device, &descriptor);
}

WGPUSampler create_decal_sampler() {
    WGPUSamplerDescriptor descriptor = {};
    descriptor.addressModeU = WGPUAddressMode_Repeat;
    descriptor.addressModeV = WGPUAddressMode_Repeat;
    descriptor.addressModeW = WGPUAddressMode_Repeat;
    descriptor.magFilter = WGPUFilterMode_Linear;
    descriptor.minFilter = WGPUFilterMode_Linear;
    descriptor.mipmapFilter = WGPUMipmapFilterMode_Linear;
    descriptor.lodMaxClamp = 32.0f;
    descriptor.maxAnisotropy = 4;
    return wgpuDeviceCreateSampler(gpu.device, &descriptor);
}

void create_depth_target(uint32_t width, uint32_t height, WGPUTexture *texture, WGPUTextureView *view) {
    WGPUTextureDescriptor descriptor = {};
    descriptor.usage = WGPUTextureUsage_RenderAttachment;
    descriptor.dimension = WGPUTextureDimension_2D;
    descriptor.size = {width, height, 1};
    descriptor.format = kDepthFormat;
    descriptor.mipLevelCount = 1;
    descriptor.sampleCount = 1;
    *texture = wgpuDeviceCreateTexture(gpu.device, &descriptor);
    *view = wgpuTextureCreateView(*texture, nullptr);
}

void release_depth_target(WGPUTexture *texture, WGPUTextureView *view) {
    if (*view) { wgpuTextureViewRelease(*view); *view = nullptr; }
    if (*texture) { wgpuTextureDestroy(*texture); wgpuTextureRelease(*texture); *texture = nullptr; }
}

// Geometria compartilhada por todos os objetos de um mesmo tipo
struct Shape {
    WGPUBuffer vertex_buffer = nullptr;
    WGPUBuffer index_buffer = nullptr;
    uint32_t index_count = 0;

    explicit Shape(const MeshData &data) {
        vertex_buffer = create_buffer_with_data(data.vertices.data(), data.vertices.size() * sizeof(float),
                                                WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst);
        index_buffer = create_buffer_with_data(data.indices.data(), data.indices.size() * sizeof(uint32_t),
                                               WGPUBufferUsage_Index | WGPUBufferUsage_CopyDst);
        index_count = static_cast<uint32_t>(data.indices.size());
    }

    void draw(WGPURenderPassEncoder pass) const {
        wgpuRenderPassEncoderSetVertexBuffer(pass, 0, vertex_buffer, 0, WGPU_WHOLE_SIZE);
        wgpuRenderPassEncoderSetIndexBuffer(pass, index_buffer, WGPUIndexFormat_Uint32, 0, WGPU_WHOLE_SIZE);
        wgpuRenderPassEncoderDrawIndexed(pass, index_count, 1, 0, 0, 0);
    }

    void release() {
        wgpuBufferRelease(vertex_buffer);
        wgpuBufferRelease(index_buffer);
    }
};

struct Texture {
    WGPUTexture texture = nullptr;
    WGPUTextureView view = nullptr;
    WGPUBindGroup bind_group = nullptr;

    explicit Texture(const Image &image) {
        uint32_t mip_count = 1;
        for (uint32_t s = std::max(image.width, image.height); s > 1; s >>= 1) ++mip_count;

        WGPUTextureDescriptor descriptor = {};
        descriptor.usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst;
        descriptor.dimension = WGPUTextureDimension_2D;
        descriptor.size = {image.width, image.height, 1};
        descriptor.format = WGPUTextureFormat_RGBA8UnormSrgb;
        descriptor.mipLevelCount = mip_count;
        descriptor.sampleCount = 1;
        texture = wgpuDeviceCreateTexture(gpu.device, &descriptor);

        std::vector<uint8_t> level = image.rgba;
        uint32_t w = image.width, h = image.height;
        for (uint32_t m = 0; m < mip_count; ++m) {
            WGPUTexelCopyTextureInfo destination = {};
            destination.texture = texture;
            destination.mipLevel = m;
            destination.aspect = WGPUTextureAspect_All;
            WGPUTexelCopyBufferLayout layout = {};
            layout.bytesPerRow = w * 4;
            layout.rowsPerImage = h;
            WGPUExtent3D extent = {w, h, 1};
            wgpuQueueWriteTexture(gpu.queue, &destination, level.data(), level.size(), &layout, &extent);

            if (m + 1 == mip_count) break;
            uint32_t nw = std::max(w / 2, 1u), nh = std::max(h / 2, 1u);
            std::vector<uint8_t> next(static_cast<size_t>(nw) * nh * 4);
            for (uint32_t y = 0; y < nh; ++y)
                for (uint32_t x = 0; x < nw; ++x)
                    for (int c = 0; c < 4; ++c) {
                        uint32_t x0 = std::min(2 * x, w - 1), x1 = std::min(2 * x + 1, w - 1);
                        uint32_t y0 = std::min(2 * y, h - 1), y1 = std::min(2 * y + 1, h - 1);
                        uint32_t sum = level[(y0 * w + x0) * 4 + c] + level[(y0 * w + x1) * 4 + c] +
                                       level[(y1 * w + x0) * 4 + c] + level[(y1 * w + x1) * 4 + c];
                        next[(static_cast<size_t>(y) * nw + x) * 4 + c] = static_cast<uint8_t>(sum / 4);
                    }
            level = std::move(next);
            w = nw;
            h = nh;
        }

        view = wgpuTextureCreateView(texture, nullptr);
        WGPUBindGroupEntry entries[2] = {};
        entries[0].binding = 0;
        entries[0].textureView = view;
        entries[1].binding = 1;
        entries[1].sampler = gpu.sampler;
        WGPUBindGroupDescriptor bind_group_descriptor = {};
        bind_group_descriptor.layout = gpu.decal_layout;
        bind_group_descriptor.entryCount = 2;
        bind_group_descriptor.entries = entries;
        bind_group = wgpuDeviceCreateBindGroup(gpu.device, &bind_group_descriptor);
    }

    void release() {
        wgpuBindGroupRelease(bind_group);
        wgpuTextureViewRelease(view);
        wgpuTextureDestroy(texture);
        wgpuTextureRelease(texture);
    }
};

// Nó de grafo de cena: transformação herdada, forma, textura e filhos.
// Nós de grupo só têm translação; a escala fica nos nós com forma.
struct Node {
    std::string name;
    Mat4 local_matrix;
    const Shape *shape = nullptr;
    const Texture *texture = nullptr;
    Vec3 color = {1.0f, 1.0f, 1.0f};
    float shininess = 32.0f;
    float uv_scale[2] = {1.0f, 1.0f};
    std::vector<std::unique_ptr<Node>> children;
    WGPUBuffer object_buffer = nullptr;
    WGPUBindGroup object_bind_group = nullptr;

    Node(std::string node_name, Mat4 local) : name(std::move(node_name)), local_matrix(local) {}

    Node *add_group(const std::string &child_name, Mat4 local) {
        children.push_back(std::make_unique<Node>(child_name, local));
        return children.back().get();
    }

    Node *add_object(const std::string &child_name, Mat4 local, const Shape *child_shape,
                     const Texture *child_texture, Vec3 child_color, float child_shininess,
                     float uv_u, float uv_v) {
        Node *child = add_group(child_name, local);
        child->shape = child_shape;
        child->texture = child_texture;
        child->color = child_color;
        child->shininess = child_shininess;
        child->uv_scale[0] = uv_u;
        child->uv_scale[1] = uv_v;
        child->object_buffer = create_uniform_buffer(sizeof(ObjectUniforms));
        child->object_bind_group =
            create_uniform_bind_group(gpu.object_layout, child->object_buffer, sizeof(ObjectUniforms));
        return child;
    }

    void prepare(const Mat4 &parent_matrix, const Mat4 &view) {
        Mat4 world = mat4_mul(parent_matrix, local_matrix);
        if (shape) {
            Mat4 modelview = mat4_mul(view, world);
            Mat4 normal_matrix = mat4_normal_matrix(modelview);
            ObjectUniforms u = {};
            std::memcpy(u.modelview, modelview.data(), sizeof(u.modelview));
            std::memcpy(u.normal_matrix, normal_matrix.data(), sizeof(u.normal_matrix));
            set4(u.base_color, color, 1.0f);
            set4(u.specular_color, {1.0f, 1.0f, 1.0f}, shininess);
            u.uv_scale[0] = uv_scale[0];
            u.uv_scale[1] = uv_scale[1];
            wgpuQueueWriteBuffer(gpu.queue, object_buffer, 0, &u, sizeof(u));
        }
        for (auto &child : children) child->prepare(world, view);
    }

    void render(WGPURenderPassEncoder pass) const {
        if (shape) {
            wgpuRenderPassEncoderSetBindGroup(pass, 1, object_bind_group, 0, nullptr);
            wgpuRenderPassEncoderSetBindGroup(pass, 2, texture->bind_group, 0, nullptr);
            shape->draw(pass);
        }
        for (const auto &child : children) child->render(pass);
    }

    void release() {
        if (object_bind_group) wgpuBindGroupRelease(object_bind_group);
        if (object_buffer) wgpuBufferRelease(object_buffer);
        for (auto &child : children) child->release();
    }
};

const char *kShaderCode = R"WGSL(
struct Light {
    position: vec4<f32>,
    direction: vec4<f32>,
    spot: vec4<f32>,
    ambient: vec4<f32>,
    diffuse: vec4<f32>,
    specular: vec4<f32>,
};

struct SceneData {
    projection: mat4x4<f32>,
    clip_plane: vec4<f32>,
    fog_color: vec4<f32>,
    fog_params: vec4<f32>,
    lights: array<Light, 2>,
};

struct ObjectData {
    modelview: mat4x4<f32>,
    normal_matrix: mat4x4<f32>,
    base_color: vec4<f32>,
    specular_color: vec4<f32>,
    uv_scale: vec4<f32>,
};

@group(0) @binding(0) var<uniform> scene: SceneData;
@group(1) @binding(0) var<uniform> obj: ObjectData;
@group(2) @binding(0) var decal_texture: texture_2d<f32>;
@group(2) @binding(1) var decal_sampler: sampler;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) eye_pos: vec3<f32>,
    @location(2) uv: vec2<f32>,
};

@vertex
fn vertex_main(input: VertexInput) -> VertexOutput {
    let p_eye = obj.modelview * vec4<f32>(input.position, 1.0);
    var output: VertexOutput;
    output.position = scene.projection * p_eye;
    output.normal = (obj.normal_matrix * vec4<f32>(input.normal, 0.0)).xyz;
    output.eye_pos = p_eye.xyz;
    output.uv = input.uv * obj.uv_scale.xy;
    return output;
}

@fragment
fn fragment_main(input: VertexOutput, @builtin(front_facing) front: bool) -> @location(0) vec4<f32> {
    let decal = textureSample(decal_texture, decal_sampler, input.uv).rgb;

    if (scene.fog_params.w > 0.5) {
        if (dot(scene.clip_plane.xyz, input.eye_pos) + scene.clip_plane.w < 0.0) {
            discard;
        }
    }

    var n = normalize(input.normal);
    if (!front) {
        n = -n;
    }
    let v = normalize(-input.eye_pos);
    let base = obj.base_color.rgb;

    var lit = vec3<f32>(0.0);
    var spec_total = vec3<f32>(0.0);

    for (var i = 0u; i < 2u; i = i + 1u) {
        let light = scene.lights[i];
        if (light.spot.w < 0.5) {
            continue;
        }
        lit += light.ambient.rgb * base;

        let l = normalize(light.position.xyz - input.eye_pos);

        var k_spot = 1.0;
        if (light.spot.z > 0.5) {
            let c = dot(-l, light.direction.xyz);
            k_spot = 0.0;
            if (c >= light.spot.x) {
                if (light.spot.y > 0.0) {
                    k_spot = pow(max(c, 0.0), light.spot.y);
                } else {
                    k_spot = 1.0;
                }
            }
        }

        let n_dot_l = dot(n, l);
        if (k_spot > 0.0 && n_dot_l > 0.0) {
            lit += k_spot * n_dot_l * light.diffuse.rgb * base;
            let r = reflect(-l, n);
            let s = pow(max(dot(r, v), 0.0), obj.specular_color.w);
            spec_total += k_spot * s * obj.specular_color.rgb * light.specular.rgb;
        }
    }

    var color = lit * decal + spec_total;

    if (scene.fog_params.z > 0.5) {
        let dist = length(input.eye_pos);
        let f = clamp((scene.fog_params.y - dist) / (scene.fog_params.y - scene.fog_params.x), 0.0, 1.0);
        color = mix(scene.fog_color.rgb, color, f);
    }
    return vec4<f32>(color, 1.0);
}
)WGSL";

} // namespace

int main() {
    if (!glfwInit()) std::exit(EXIT_FAILURE);
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow *window = glfwCreateWindow(800, 600, "Projeto 2 - Cena 3D", nullptr, nullptr);
    if (!window) std::exit(EXIT_FAILURE);

    WGPUInstance instance = wgpuCreateInstance(nullptr);
    WGPUSurface surface = glfwCreateWindowWGPUSurface(instance, window);

    WGPURequestAdapterOptions adapter_options = {};
    adapter_options.powerPreference = WGPUPowerPreference_HighPerformance;
    adapter_options.compatibleSurface = surface;
    AdapterRequest adapter_request;
    WGPURequestAdapterCallbackInfo adapter_callback_info = {};
    adapter_callback_info.mode = WGPUCallbackMode_AllowProcessEvents;
    adapter_callback_info.callback = handle_request_adapter;
    adapter_callback_info.userdata1 = &adapter_request;
    wgpuInstanceRequestAdapter(instance, &adapter_options, adapter_callback_info);
    wgpuInstanceProcessEvents(instance);
    WGPUAdapter adapter = adapter_request.adapter;
    if (!adapter) { std::fprintf(stderr, "Falha ao obter adapter\n"); std::exit(EXIT_FAILURE); }

    DeviceRequest device_request;
    WGPURequestDeviceCallbackInfo device_callback_info = {};
    device_callback_info.mode = WGPUCallbackMode_AllowProcessEvents;
    device_callback_info.callback = handle_request_device;
    device_callback_info.userdata1 = &device_request;
    wgpuAdapterRequestDevice(adapter, nullptr, device_callback_info);
    wgpuInstanceProcessEvents(instance);
    gpu.device = device_request.device;
    if (!gpu.device) { std::fprintf(stderr, "Falha ao obter device\n"); std::exit(EXIT_FAILURE); }
    gpu.queue = wgpuDeviceGetQueue(gpu.device);

    WGPUSurfaceCapabilities surface_capabilities = {};
    wgpuSurfaceGetCapabilities(surface, adapter, &surface_capabilities);
    WGPUTextureFormat surface_format = surface_capabilities.formats[0];
    for (size_t i = 0; i < surface_capabilities.formatCount; ++i) {
        WGPUTextureFormat f = surface_capabilities.formats[i];
        if (f == WGPUTextureFormat_BGRA8UnormSrgb || f == WGPUTextureFormat_RGBA8UnormSrgb) {
            surface_format = f;
            break;
        }
    }

    WGPUSurfaceConfiguration surface_config = {};
    surface_config.device = gpu.device;
    surface_config.format = surface_format;
    surface_config.usage = WGPUTextureUsage_RenderAttachment;
    surface_config.presentMode = WGPUPresentMode_Fifo;
    surface_config.alphaMode = surface_capabilities.alphaModes[0];
    {
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        surface_config.width = static_cast<uint32_t>(width);
        surface_config.height = static_cast<uint32_t>(height);
    }
    wgpuSurfaceConfigure(surface, &surface_config);

    WGPUTexture depth_texture = nullptr;
    WGPUTextureView depth_view = nullptr;
    create_depth_target(surface_config.width, surface_config.height, &depth_texture, &depth_view);

    WGPUBindGroupLayout global_layout = create_uniform_layout(sizeof(GlobalUniforms));
    gpu.object_layout = create_uniform_layout(sizeof(ObjectUniforms));
    gpu.decal_layout = create_decal_layout();
    gpu.sampler = create_decal_sampler();

    WGPUBuffer global_buffer = create_uniform_buffer(sizeof(GlobalUniforms));
    WGPUBindGroup global_bind_group = create_uniform_bind_group(global_layout, global_buffer, sizeof(GlobalUniforms));

    WGPUShaderSourceWGSL wgsl_source = {};
    wgsl_source.chain.sType = WGPUSType_ShaderSourceWGSL;
    wgsl_source.code = {kShaderCode, WGPU_STRLEN};
    WGPUShaderModuleDescriptor shader_descriptor = {};
    shader_descriptor.nextInChain = &wgsl_source.chain;
    WGPUShaderModule shader = wgpuDeviceCreateShaderModule(gpu.device, &shader_descriptor);

    WGPUBindGroupLayout bind_group_layouts[3] = {global_layout, gpu.object_layout, gpu.decal_layout};
    WGPUPipelineLayoutDescriptor pipeline_layout_descriptor = {};
    pipeline_layout_descriptor.bindGroupLayoutCount = 3;
    pipeline_layout_descriptor.bindGroupLayouts = bind_group_layouts;
    WGPUPipelineLayout pipeline_layout = wgpuDeviceCreatePipelineLayout(gpu.device, &pipeline_layout_descriptor);

    std::array<WGPUVertexAttribute, 3> attributes = {};
    attributes[0] = {WGPUVertexFormat_Float32x3, 0, 0};
    attributes[1] = {WGPUVertexFormat_Float32x3, 3 * sizeof(float), 1};
    attributes[2] = {WGPUVertexFormat_Float32x2, 6 * sizeof(float), 2};

    WGPUVertexBufferLayout vertex_buffer_layout = {};
    vertex_buffer_layout.arrayStride = 8 * sizeof(float);
    vertex_buffer_layout.stepMode = WGPUVertexStepMode_Vertex;
    vertex_buffer_layout.attributeCount = attributes.size();
    vertex_buffer_layout.attributes = attributes.data();

    WGPUColorTargetState color_target = {};
    color_target.format = surface_format;
    color_target.writeMask = WGPUColorWriteMask_All;

    WGPUFragmentState fragment_state = {};
    fragment_state.module = shader;
    fragment_state.entryPoint = {"fragment_main", WGPU_STRLEN};
    fragment_state.targetCount = 1;
    fragment_state.targets = &color_target;

    WGPUDepthStencilState depth_stencil = {};
    depth_stencil.format = kDepthFormat;
    depth_stencil.depthWriteEnabled = WGPUOptionalBool_True;
    depth_stencil.depthCompare = WGPUCompareFunction_Less;
    depth_stencil.stencilFront.compare = WGPUCompareFunction_Always;
    depth_stencil.stencilFront.failOp = WGPUStencilOperation_Keep;
    depth_stencil.stencilFront.depthFailOp = WGPUStencilOperation_Keep;
    depth_stencil.stencilFront.passOp = WGPUStencilOperation_Keep;
    depth_stencil.stencilBack = depth_stencil.stencilFront;
    depth_stencil.stencilReadMask = 0xFFFFFFFF;
    depth_stencil.stencilWriteMask = 0xFFFFFFFF;

    // sem culling: o interior dos objetos cortados precisa aparecer
    WGPURenderPipelineDescriptor pipeline_descriptor = {};
    pipeline_descriptor.layout = pipeline_layout;
    pipeline_descriptor.vertex.module = shader;
    pipeline_descriptor.vertex.entryPoint = {"vertex_main", WGPU_STRLEN};
    pipeline_descriptor.vertex.bufferCount = 1;
    pipeline_descriptor.vertex.buffers = &vertex_buffer_layout;
    pipeline_descriptor.primitive.topology = WGPUPrimitiveTopology_TriangleList;
    pipeline_descriptor.primitive.frontFace = WGPUFrontFace_CCW;
    pipeline_descriptor.primitive.cullMode = WGPUCullMode_None;
    pipeline_descriptor.depthStencil = &depth_stencil;
    pipeline_descriptor.multisample.count = 1;
    pipeline_descriptor.multisample.mask = 0xFFFFFFFF;
    pipeline_descriptor.fragment = &fragment_state;
    WGPURenderPipeline pipeline = wgpuDeviceCreateRenderPipeline(gpu.device, &pipeline_descriptor);

    Shape cube(make_cube());
    Shape sphere(make_sphere(32, 64));
    Texture wood(make_wood(256));
    Texture checker(make_checker(256, 8));
    Texture stripes(make_stripes(256, 8));
    Texture white(make_white());

    const Vec3 table_color = {0.85f, 0.85f, 0.85f};
    const float leg_height = 0.94f;

    Node root("raiz", mat4_identity());
    Node *table = root.add_group("mesa", mat4_identity());
    table->add_object("tampo", translate_scale({0.0f, 1.0f, 0.0f}, {3.0f, 0.12f, 2.0f}),
                      &cube, &wood, table_color, 16.0f, 2.0f, 3.0f);
    for (int sx = -1; sx <= 1; sx += 2)
        for (int sz = -1; sz <= 1; sz += 2)
            table->add_object("pé", translate_scale({1.35f * sx, leg_height * 0.5f, 0.85f * sz},
                                                    {0.15f, leg_height, 0.15f}),
                              &cube, &wood, table_color, 16.0f, 1.0f, 1.0f);

    Node *box = table->add_group("caixa", mat4_translate({-0.4f, 1.26f, 0.0f}));
    box->add_object("cubo da caixa", translate_scale({0.0f, 0.0f, 0.0f}, {0.8f, 0.4f, 0.6f}),
                    &cube, &checker, {0.90f, 0.70f, 0.20f}, 32.0f, 2.0f, 2.0f);
    box->add_object("esfera verde", translate_scale({0.0f, 0.45f, 0.0f}, {0.25f, 0.25f, 0.25f}),
                    &sphere, &white, {0.55f, 0.85f, 0.45f}, 64.0f, 1.0f, 1.0f);

    table->add_object("esfera vermelha", translate_scale({0.9f, 1.46f, -0.3f}, {0.4f, 0.4f, 0.4f}),
                      &sphere, &stripes, {1.0f, 0.35f, 0.35f}, 64.0f, 1.0f, 1.0f);

    const Vec3 eye = {3.0f, 3.5f, 5.0f};
    const Vec3 center = {0.0f, 1.0f, 0.0f};
    const Mat4 look_at = mat4_look_at(eye, center, {0.0f, 1.0f, 0.0f});

    SpotLight spot_light;
    spot_light.position = {1.0f, 4.0f, 2.0f};
    spot_light.direction = normalize(Vec3{0.25f, 1.3f, -0.1f} - spot_light.position);
    spot_light.angle = 22.0f * kPi / 180.0f;
    spot_light.exponent = 15.0f;
    spot_light.ambient = {0.08f, 0.08f, 0.08f};

    PointLight point_light;
    point_light.position = {-2.0f, 2.5f, 2.0f};
    point_light.diffuse = {0.25f, 0.35f, 0.70f};
    point_light.specular = {0.30f, 0.40f, 0.80f};

    const Vec3 fog_color = {0.12f, 0.14f, 0.22f};
    const float fog_start = 5.0f, fog_end = 10.0f;
    const Vec3 clip_normal = {0.0f, -1.0f, 0.0f};
    const float clip_d = 1.55f;

    App app(length(eye - center));
    glfwSetWindowUserPointer(window, &app);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_pos_callback);
    glfwSetKeyCallback(window, key_callback);
    std::printf("Mouse: arcball | F: neblina | C: plano de corte | L: segunda luz | Esc: sair\n");

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        if (width == 0 || height == 0) continue;

        if (static_cast<uint32_t>(width) != surface_config.width ||
            static_cast<uint32_t>(height) != surface_config.height) {
            surface_config.width = static_cast<uint32_t>(width);
            surface_config.height = static_cast<uint32_t>(height);
            wgpuSurfaceConfigure(surface, &surface_config);
            release_depth_target(&depth_texture, &depth_view);
            create_depth_target(surface_config.width, surface_config.height, &depth_texture, &depth_view);
        }

        Mat4 view = mat4_mul(app.arcball.mat, look_at);
        Mat4 projection = mat4_perspective(45.0f * kPi / 180.0f,
                                           static_cast<float>(width) / static_cast<float>(height), 0.1f, 50.0f);

        point_light.enabled = app.light2;
        GlobalUniforms global = {};
        std::memcpy(global.projection, projection.data(), sizeof(global.projection));
        spot_light.write_fields(global.lights[0], view);
        point_light.write_fields(global.lights[1], view);

        // a view só tem rotação R e translação t: n' = R n, d' = d - n'.t
        Vec3 clip_n = transform_direction(view, clip_normal);
        float clip_d_eye = clip_d - dot(clip_n, Vec3{view[12], view[13], view[14]});
        set4(global.clip_plane, clip_n, clip_d_eye);

        set4(global.fog_color, fog_color, 1.0f);
        global.fog_params[0] = fog_start;
        global.fog_params[1] = fog_end;
        global.fog_params[2] = app.fog ? 1.0f : 0.0f;
        global.fog_params[3] = app.clip ? 1.0f : 0.0f;
        wgpuQueueWriteBuffer(gpu.queue, global_buffer, 0, &global, sizeof(global));

        root.prepare(mat4_identity(), view);

        WGPUSurfaceTexture surface_texture = {};
        wgpuSurfaceGetCurrentTexture(surface, &surface_texture);
        if (surface_texture.texture == nullptr) continue;

        WGPUTextureView color_view = wgpuTextureCreateView(surface_texture.texture, nullptr);
        WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(gpu.device, nullptr);

        WGPURenderPassColorAttachment color_attachment = {};
        color_attachment.view = color_view;
        color_attachment.depthSlice = WGPU_DEPTH_SLICE_UNDEFINED;
        color_attachment.loadOp = WGPULoadOp_Clear;
        color_attachment.storeOp = WGPUStoreOp_Store;
        color_attachment.clearValue = {fog_color.x, fog_color.y, fog_color.z, 1.0};

        WGPURenderPassDepthStencilAttachment depth_attachment = {};
        depth_attachment.view = depth_view;
        depth_attachment.depthLoadOp = WGPULoadOp_Clear;
        depth_attachment.depthStoreOp = WGPUStoreOp_Store;
        depth_attachment.depthClearValue = 1.0f;

        WGPURenderPassDescriptor render_pass_descriptor = {};
        render_pass_descriptor.colorAttachmentCount = 1;
        render_pass_descriptor.colorAttachments = &color_attachment;
        render_pass_descriptor.depthStencilAttachment = &depth_attachment;

        WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(encoder, &render_pass_descriptor);
        wgpuRenderPassEncoderSetPipeline(pass, pipeline);
        wgpuRenderPassEncoderSetBindGroup(pass, 0, global_bind_group, 0, nullptr);
        root.render(pass);
        wgpuRenderPassEncoderEnd(pass);
        wgpuRenderPassEncoderRelease(pass);

        WGPUCommandBuffer command_buffer = wgpuCommandEncoderFinish(encoder, nullptr);
        wgpuQueueSubmit(gpu.queue, 1, &command_buffer);
        wgpuSurfacePresent(surface);

        wgpuCommandBufferRelease(command_buffer);
        wgpuCommandEncoderRelease(encoder);
        wgpuTextureViewRelease(color_view);
        wgpuTextureRelease(surface_texture.texture);
    }

    root.release();
    wood.release();
    checker.release();
    stripes.release();
    white.release();
    cube.release();
    sphere.release();
    release_depth_target(&depth_texture, &depth_view);
    wgpuSamplerRelease(gpu.sampler);

    wgpuRenderPipelineRelease(pipeline);
    wgpuPipelineLayoutRelease(pipeline_layout);
    wgpuBindGroupRelease(global_bind_group);
    wgpuBufferRelease(global_buffer);
    wgpuBindGroupLayoutRelease(gpu.decal_layout);
    wgpuBindGroupLayoutRelease(gpu.object_layout);
    wgpuBindGroupLayoutRelease(global_layout);
    wgpuShaderModuleRelease(shader);
    wgpuSurfaceCapabilitiesFreeMembers(surface_capabilities);
    wgpuQueueRelease(gpu.queue);
    wgpuDeviceRelease(gpu.device);
    wgpuAdapterRelease(adapter);
    wgpuSurfaceRelease(surface);
    glfwDestroyWindow(window);
    wgpuInstanceRelease(instance);
    glfwTerminate();
    return 0;
}