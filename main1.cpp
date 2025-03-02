//
// Created by Ninter6 on 2025/2/22.
//

#include <iostream>
#include <fstream>
#include <cctype>
#include <cassert>
#include <map>
#include <span>
#include <optional>
#include <algorithm>
#include <unordered_map>

#include "glad/glad.h"
#include <GLFW/glfw3.h>

#include "json.hpp"

#include "mathpls.h"

using namespace mathpls;

constexpr ivec2 window_size = {1440, 900};

constexpr float vertices[] = {
    0.0f,0.0f,1.0f, 1.0f,0.0f,1.0f, 1.0f,1.0f,1.0f,
    0.0f,0.0f,1.0f, 1.0f,1.0f,1.0f, 0.0f,1.0f,1.0f,

    1.0f,0.0f,0.0f, 0.0f,0.0f,0.0f, 0.0f,1.0f,0.0f,
    1.0f,0.0f,0.0f, 0.0f,1.0f,0.0f, 1.0f,1.0f,0.0f,

    0.0f,1.0f,1.0f, 1.0f,1.0f,1.0f, 1.0f,1.0f,0.0f,
    0.0f,1.0f,1.0f, 1.0f,1.0f,0.0f, 0.0f,1.0f,0.0f,

    0.0f,0.0f,0.0f, 1.0f,0.0f,0.0f, 1.0f,0.0f,1.0f,
    0.0f,0.0f,0.0f, 1.0f,0.0f,1.0f, 0.0f,0.0f,1.0f,

    1.0f,0.0f,0.0f, 1.0f,1.0f,0.0f, 1.0f,1.0f,1.0f,
    1.0f,0.0f,0.0f, 1.0f,1.0f,1.0f, 1.0f,0.0f,1.0f,

    0.0f,0.0f,1.0f, 0.0f,1.0f,1.0f, 0.0f,1.0f,0.0f,
    0.0f,0.0f,1.0f, 0.0f,1.0f,0.0f, 0.0f,0.0f,0.0f,

    1.0f,0.0f,0.0f, 1.0f,0.0f,0.0f, 1.0f,0.0f,0.0f,
    1.0f,0.0f,0.0f, 1.0f,0.0f,0.0f, 1.0f,0.0f,0.0f,

    1.0f,.65f,0.0f, 1.0f,.65f,0.0f, 1.0f,.65f,0.0f,
    1.0f,.65f,0.0f, 1.0f,.65f,0.0f, 1.0f,.65f,0.0f,

    1.0f,1.0f,1.0f, 1.0f,1.0f,1.0f, 1.0f,1.0f,1.0f,
    1.0f,1.0f,1.0f, 1.0f,1.0f,1.0f, 1.0f,1.0f,1.0f,

    1.0f,1.0f,0.0f, 1.0f,1.0f,0.0f, 1.0f,1.0f,0.0f,
    1.0f,1.0f,0.0f, 1.0f,1.0f,0.0f, 1.0f,1.0f,0.0f,

    0.0f,0.0f,1.0f, 0.0f,0.0f,1.0f, 0.0f,0.0f,1.0f,
    0.0f,0.0f,1.0f, 0.0f,0.0f,1.0f, 0.0f,0.0f,1.0f,

    0.0f,0.5f,0.0f, 0.0f,0.5f,0.0f, 0.0f,0.5f,0.0f,
    0.0f,0.5f,0.0f, 0.0f,0.5f,0.0f, 0.0f,0.5f,0.0f,
};

constexpr auto vsh = R"(
#version 410 core
layout (location = 0) in vec3 pos;
layout (location = 5) in vec3 col;
layout (location = 1) in vec4 m0;
layout (location = 2) in vec4 m1;
layout (location = 3) in vec4 m2;
layout (location = 4) in vec4 m3;

uniform mat4 proj;
uniform mat4 view;

out vec4 f_col;

void main() {
    mat4 model = mat4(m0, m1, m2, m3);
    gl_Position = proj * view * model * vec4(pos, 1.0);
    f_col = vec4(col, 1.0);
})";

constexpr auto fsh = R"(
#version 410 core
in vec4 f_col;
out vec4 color;
void main() {
    color = f_col;
})";

GLFWwindow* window;
GLuint vao, cube_vbo, m_vbo, sh;
mat4* m_map = nullptr;
int m_count = 0;

vec3 cam = {0, 0, 3};

void init_window() {
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        exit(EXIT_FAILURE);
    }

    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_SAMPLES, 4);

#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    const ivec2 ext = window_size/2;
#else
    const ivec2 ext = window_size;
#endif

    window = glfwCreateWindow(ext.x, ext.y, "rigid animation", nullptr, nullptr);
    assert(window && "Failed to create window");

    glfwMakeContextCurrent(window);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD" << std::endl;
        std::terminate();
    }

    glViewport(0, 0, window_size.x, window_size.y);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);

    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    glGenBuffers(1, &cube_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, cube_vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(5);
    glVertexAttribPointer(5, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)(sizeof(float) * 108));

    glGenBuffers(1, &m_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(mat4) * 256, nullptr, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(mat4), (void*)0);
    glVertexAttribDivisor(1, 1);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(mat4), (void*)0);
    glVertexAttribDivisor(1, 1);
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, sizeof(mat4), (void*)(sizeof(float)*4));
    glVertexAttribDivisor(2, 1);
    glEnableVertexAttribArray(3);
    glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, sizeof(mat4), (void*)(sizeof(float)*8));
    glVertexAttribDivisor(3, 1);
    glEnableVertexAttribArray(4);
    glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, sizeof(mat4), (void*)(sizeof(float)*12));
    glVertexAttribDivisor(4, 1);
    m_map = (mat4*)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);

    GLuint vert = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vert, 1, &vsh, nullptr);
    glCompileShader(vert);
    GLuint frag = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(frag, 1, &fsh, nullptr);
    glCompileShader(frag);
    sh = glCreateProgram();
    glAttachShader(sh, vert);
    glAttachShader(sh, frag);
    glLinkProgram(sh);
    glDeleteShader(vert);
    glDeleteShader(frag);

    GLint success;
    glGetProgramiv(sh, GL_LINK_STATUS, &success);
    if (!success) std::terminate();

    glUseProgram(sh);
    glUniformMatrix4fv(glGetUniformLocation(sh, "proj"), 1, GL_FALSE,
        perspective(radians(45.f), (float)window_size.x / window_size.y, 0.1f, 100.0f).value_ptr());
}

void process_input() {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        cam = quat{cross(vec3{0, 1, 0}, cam).normalized(), .1f} * cam;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        cam = quat{cross(cam, vec3{0, 1, 0}).normalized(), .1f} * cam;
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        cam = quat{{0, 1, 0}, .1f} * cam;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        cam = quat{{0,-1, 0}, .1f} * cam;
    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
        cam = cam.normalized() * (cam.length() + .1f);
    if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
        cam = cam.normalized() * (cam.length() - .1f);
}

struct Cube {
    vec3 pos{}; // position of centre
    vec3 ext{1}; // extent of the cube
    vec3 cnt{.5}; // centre of rotation
    quat rot{}; // rotation

    [[nodiscard]] mat4 matrix() const {
        auto ts = mat4::zero();
        ts[0][0] = ext.x;
        ts[1][1] = ext.y;
        ts[2][2] = ext.z;
        ts[3] = vec4{-cnt, 1.f};
        return local() * ts;
    }
    [[nodiscard]] mat4 local() const {
        auto tr = rotate(rot);
        tr[3] = vec4{pos, 1.f};
        return tr;
    }
};

void add_cube(const Cube& c) {
    m_map[m_count++] = c.matrix();
}

void clear_cube() {
    m_count = 0;
}

void draw_cube() {
    glBindVertexArray(vao);
    glDrawArraysInstanced(GL_TRIANGLES, 0, 36, m_count);
}

uint32_t s2id(std::string_view s) {
    int gen = std::toupper(s[0]) - 'K';
    int id = std::stoi({s.substr(1).data(), s.size() - 1});
    return (gen << 16) | id;
}

vec3 j2vec3(const nlohmann::json& j) {
    assert(j.is_array());
    return {j[0], j[1], j[2]};
}
quat j2quat(const nlohmann::json& j) {
    // json stores as rotation vector
    auto v = j2vec3(j);
    auto l = v.length();
    return l < 1e-4f ? quat{} : quat{v / l, l};
}

struct Joint {
    vec3 pos;
    quat rot;
};

template <class T>
struct key_unit {
    T v, m;
    float t;
};

using key_vec3 = key_unit<vec3>;
using key_quat = key_unit<quat>;

struct Model {
    struct cube_info { uint32_t cur, par; };

    Model() = default;

    size_t find(uint32_t id);

    std::vector<Cube> cubes;
    std::vector<cube_info> info;
};

size_t Model::find(uint32_t id) {
    auto i = std::lower_bound(info.begin(), info.end(), id, [](auto&& a, auto&& b) {
        return a.cur < b;
    }) - info.begin();
    return i != info.size() && info[i].cur == id ? i : SIZE_MAX;
}

struct axis_info {
    uint32_t pos_b, pos_n;
    uint32_t rot_b, rot_n;
    bool pos_flat, rot_flat;
};

namespace algo {

vec3 ln(const quat& q) {
    auto& [w, x, y, z] = q.asArray;
    return vec3(x, y, z)*(acos(w)/sqrt(1-w*w));
}
quat exp(const vec3& v) {
    auto l = v.length();
    if (l < 1e-4f) return {};
    return {cos(l), v*(sin(l)/l)};
}

quat quat_grad(const quat& q0, const quat& q1, const quat& q2) {
    auto q1s = q1.conjugate();
    auto lnq1sq0 = ln(q1s*q0);
    auto lnq1sq2 = ln(q1s*q2);
    return q1 * exp((lnq1sq0+lnq1sq2) * -.25f);
}

vec3 vec_grad(const vec3& v0, const vec3& v1, const vec3& v2, float t0, float t1, float t2) {
    auto d0 = (v1 - v0) / (t1 - t0);
    auto d1 = (v2 - v1) / (t2 - t1);
    return lerp(d0, d1, (t1 - t0) / (t2 - t0));
}

vec3 vec3_flat(const vec3& v0, const vec3& v1, float t) {
    return lerp(v0, v1, t);
}

quat quat_flat(const quat& q0, const quat& q1, float t) {
    return slerp(q0, q1, t);
}

vec3 vec3_smooth(const vec3& v0, const vec3& v1, const vec3& m0, const vec3& m1, float t) {
    const auto t3 = t*t*t, t2 = t*t;
    const auto h00 = 2*t3 - 3*t2 + 1;
    const auto h01 = 3*t2 - 2*t3;
    const auto h10 = t3 - 2*t2 + t;
    const auto h11 = t3 - t2;
    return v0*h00 + v1*h01 + m0*h10 + m1*h11;
}

quat quat_smooth(const quat& q0, const quat& q1, const quat& m0, const quat& m1, float t) {
    auto a = slerp(q0, q1, t);
    auto b = slerp(m0, m1, t);
    return slerp(a, b, 2*t*(1-t));
}

}

struct Animation {
    enum play_mode : uint8_t
    { repeat, clamp };

    void calcu_grad();
    void calcu_grad(uint32_t id);

    void apply_to_model(float t, Model& m) const;

    [[nodiscard]] vec3 pos_at(uint32_t id, float t) const;
    [[nodiscard]] quat rot_at(uint32_t id, float t) const;

    std::unordered_map<uint32_t, axis_info> info;
    std::vector<key_vec3> pos;
    std::vector<key_quat> rot;
    float duration = 1.f;
    play_mode mode = repeat;
};

void Animation::calcu_grad() {
    for (auto& [id, _] : info)
        calcu_grad(id);
}

void Animation::calcu_grad(uint32_t id) {
    const auto& f = info[id];
    auto pb = f.pos_b, pe = pb + f.pos_n;
    auto rb = f.rot_b, re = rb + f.rot_n;
    if (!f.pos_flat) for (auto i = pb+1; i+1 < pe; ++i)
        pos[i].m = algo::vec_grad(pos[i-1].v, pos[i].v, pos[i+1].v, pos[i-1].t, pos[i].t, pos[i+1].t);
    if (!f.rot_flat) for (auto i = rb+1; i+1 < re; ++i)
        rot[i].m = algo::quat_grad(rot[i-1].v, rot[i].v, rot[i+1].v);
    if (mode == repeat) {
        // assume the last is the same as the first
        if (!f.pos_flat && f.pos_n > 2)
            pos[pe-1].m = pos[pb].m = algo::vec_grad(pos[pe-2].v, pos[pb].v, pos[pb+1].v, pos[pe-2].t-duration, pos[pb].t, pos[pb+1].t);
        if (!f.rot_flat && f.rot_n > 2)
            rot[re-1].m = rot[rb].m = algo::quat_grad(rot[re-2].v, rot[rb].v, rot[rb+1].v);
    }
}

vec3 Animation::pos_at(uint32_t id, float t) const {
    assert(info.contains(id) && info.at(id).pos_n > 0);
    if (t < 0 || t > duration)
        switch (mode) {
            case repeat: t = fmodf(t, duration); break;
            case clamp: t = std::clamp(t, 0.f, duration); break;
            default: throw std::runtime_error("Invalid play mode");
        }

    auto& f = info.at(id);
    auto b = pos.begin() + f.pos_b, e = b + f.pos_n;
    auto i = std::lower_bound(b, e, t, [](const key_vec3& k, float t) {
        return k.t < t;
    });
    if (i == b || i->t - t < 1e-4f) return i->v;
    if (i == e) return (i-1)->v;

    auto& [v0, m0, t0] = *(i-1);
    auto& [v1, m1, t1] = *i;
    return f.pos_flat ?
        algo::vec3_flat(v0, v1, (t - t0) / (t1 - t0)) :
        algo::vec3_smooth(v0, v1, m0, m1, (t - t0) / (t1 - t0));
}

quat Animation::rot_at(uint32_t id, float t) const {
    assert(info.contains(id) && info.at(id).rot_n > 0);
    if (t < 0 || t > duration)
        switch (mode) {
            case repeat: t = fmodf(t, duration); break;
            case clamp: t = std::clamp(t, 0.f, duration); break;
            default: throw std::runtime_error("Invalid play mode");
        }

    auto& f = info.at(id);
    auto b = rot.begin() + f.rot_b, e = b + f.rot_n;
    auto i = std::lower_bound(b, e, t, [](const key_quat& k, float t) {
        return k.t < t;
    });
    if (i == b || i->t - t < 1e-4f) return i->v;
    if (i == e) return (i-1)->v;

    auto& [v0, m0, t0] = *(i-1);
    auto& [v1, m1, t1] = *i;

    return f.rot_flat ?
        algo::quat_flat(v0, v1, (t - t0) / (t1 - t0)) :
        algo::quat_smooth(v0, v1, m0, m1, (t - t0) / (t1 - t0));
}

void Animation::apply_to_model(float t, Model& m) const {
    assert(m.cubes.size() == m.info.size());
    for (int i = 0; i < m.info.size(); ++i) {
        auto [cur, par] = m.info[i];
        auto& c = m.cubes[i];
        if (auto it = info.find(cur); it != info.end()) {
            auto& f = it->second;
            if (f.pos_n > 0) c.pos = pos_at(cur, t);
            if (f.rot_n > 0) c.rot = rot_at(cur, t);
        }
        if (par != -1) {
            auto& p = m.cubes[m.find(par)];
            auto tr = p.local() * c.local();
            c.pos = tr[3];
            c.rot = quat_cast(tr);
        }
    }
}

struct Bone {
    // std::unordered_map<uint32_t, Joint> joints;
    std::unordered_map<std::string, Animation> anims;
};

struct anim_manager {
    anim_manager() = default;

    void load(std::string_view filename); // mixed file
    void load_model(std::string_view filename);
    void load_bone(std::string_view filename);

    void get_draw_data();

    std::unordered_map<std::string, Model> models;
    std::unordered_map<std::string, Bone> bones;

private:

    void parse_model(const nlohmann::json& j);
    void parse_bone(const nlohmann::json& j);

};

void anim_manager::load_model(std::string_view file) {
    std::ifstream ifs{{file.data(), file.size()}};
    auto json = nlohmann::json::parse(ifs);

    if (json.is_array())
        for (auto&& item : json)
            parse_model(item);
    else parse_model(json);
}

void anim_manager::load_bone(std::string_view file) {
    std::ifstream ifs{{file.data(), file.size()}};
    auto json = nlohmann::json::parse(ifs);

    if (json.is_array())
        for (auto&& item : json)
            parse_bone(item);
    else parse_bone(json);
}

void anim_manager::parse_model(const nlohmann::json& j) {
    assert(j.is_object() && j.contains("name"));
    assert(!j.contains("type") || j["type"] == "model");

    auto [it, succ] = models.try_emplace(j["name"].get<std::string>());
    if (!succ) {
        std::cerr << "Duplicate model: " << j["name"] << std::endl;
        return;
    }

    auto& model = it->second;
    std::map<uint32_t, std::pair<Model::cube_info, Cube>> cubes;
    assert(j["cubes"].is_object());
    for (auto&& [k, v] : j["cubes"].items()) {
        assert(v.is_object());
        auto& [i, c] = cubes[s2id(k)];
        i = {s2id(k), v.contains("parent") ?
            s2id(v["parent"].get<std::string>()) : -1};
        if (v.contains("pos")) c.pos = j2vec3(v["pos"]);
        if (v.contains("ext")) c.ext = j2vec3(v["ext"]);
        if (v.contains("cnt")) c.cnt = j2vec3(v["cnt"]);
        if (v.contains("rot")) c.rot = j2quat(v["rot"]);
    }
    for (auto&& [_, p] : cubes) {
        model.info.push_back(p.first);
        model.cubes.push_back(p.second);
    }
}

void anim_manager::parse_bone(const nlohmann::json& j) {
    assert(j.is_object() && j.contains("name"));
    assert(!j.contains("type") || j["type"] == "bone");

    auto [it, succ] = bones.try_emplace(j["name"].get<std::string>());
    if (!succ) {
        std::cerr << "Duplicate bone: " << j["name"] << std::endl;
        return;
    }

    auto& anims = it->second.anims;
    for (auto&& [k, v] : j["anims"].items()) {
        auto [it, succ] = anims.try_emplace(k);
        if (!succ) {
            std::cerr << "Duplicate animation: " << k << " in bone: " << j["name"] << std::endl;
            continue;
        }

        auto& a = it->second;
        assert(v.is_object() && v.contains("duration"));
        a.duration = v["duration"];
        if (v.contains("mode"))
            a.mode = v["mode"] == "repeat" ? Animation::repeat : Animation::clamp;

        for (auto&& [k, u] : v["axes"].items()) {
            auto [it, succ] = a.info.try_emplace(s2id(k));
            if (!succ) {
                std::cerr << "Duplicate axis in animation: " << k << " in bone: " << j["name"] << std::endl;
                continue;
            }

            auto& info = it->second;
            assert(u.is_object());
            if (u.contains("pos")) {
                auto&& upos = u["pos"];
                assert(upos.is_object());
                assert(upos.contains("keys") && upos["keys"].is_array());
                auto& keys = upos["keys"];
                info.pos_n = keys.size();
                if (upos.contains("flat")) info.pos_flat = upos["flat"];
                if (info.pos_n == 0) continue;
                info.pos_b = a.pos.size();
                std::transform(keys.begin(), keys.end(), std::back_inserter(a.pos), [](auto&& p) {
                    return key_vec3{ .v = j2vec3(p["v"]), .t = p["t"] };
                });
                std::stable_sort(a.pos.data() + info.pos_b, a.pos.data() + info.pos_b + info.pos_n, [](auto&& a, auto&& b) {
                    return a.t < b.t;
                });
            }
            if (u.contains("rot")) {
                auto&& urot = u["rot"];
                assert(urot.is_object());
                assert(urot.contains("keys") && urot["keys"].is_array());
                auto& keys = urot["keys"];
                info.rot_n = keys.size();
                if (urot.contains("flat")) info.rot_flat = urot["flat"];
                if (info.rot_n == 0) continue;
                info.rot_b = a.rot.size();
                std::transform(keys.begin(), keys.end(), std::back_inserter(a.rot), [](auto&& p) {
                    return key_quat{ .v = j2quat(p["v"]), .t = p["t"] };
                });
                std::stable_sort(a.rot.data() + info.rot_b, a.rot.data() + info.rot_b + info.rot_n, [](auto&& a, auto&& b) {
                    return a.t < b.t;
                });
            }
        }
        a.calcu_grad();
    }
}

void add_model(const Model& m) {
    for (auto&& c : m.cubes)
        add_cube(c);
}

int main() {
    init_window();

    anim_manager anim;
    anim.load_model(FILE_ROOT"test.json");
    anim.load_bone(FILE_ROOT"man.json");

    while (!glfwWindowShouldClose(window)) {
        glClearColor(.2, .3, .3, 1.);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        process_input();

        glUseProgram(sh);
        glUniformMatrix4fv(glGetUniformLocation(sh, "view"), 1, GL_FALSE,
            lookAt(cam, vec3{0}, vec3{0, 1, 0}).value_ptr());

        auto m = anim.models["test"];
        anim.bones["man"].anims["run"].apply_to_model(glfwGetTime(), m);

        add_model(m);
        draw_cube();
        clear_cube();

        glfwSwapBuffers(window);
        glfwPollEvents();
    }
}