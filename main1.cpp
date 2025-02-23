//
// Created by Ninter6 on 2025/2/22.
//

#include <iostream>
#include <fstream>
#include <cctype>
#include <cassert>
#include <unordered_map>
#include <map>
#include <span>

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

    mat4 matrix() const {
        auto ts = mat4::zero();
        ts[0][0] = ext.x;
        ts[1][1] = ext.y;
        ts[2][2] = ext.z;
        ts[3] = vec4{-cnt, 1.f};
        return local() * ts;
    }
    mat4 local() const {
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

    std::unordered_map<uint32_t, Cube> cubes;
};

struct axis_info {
    uint32_t pos_b, pos_n;
    uint32_t rot_b, rot_n;
    bool pos_flat, rot_flat;
};

namespace algo {

constexpr auto ln(const quat& q) {
    auto& [w, x, y, z] = q.asArray;
    return vec3(x, y, z)*(acos(w)/sqrt(1-w*w));
}
constexpr quat exp(const vec3& v) {
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

}

struct Animation {
    enum play_mode : uint8_t
    { repeat, clamp };

    void calcu_grad();
    void calcu_grad(uint32_t id);

    void apply(float t,
        const std::function<bool(uint32_t)>& need_pos,
        const std::function<bool(uint32_t)>& need_rot,
        const std::function<void(uint32_t, const vec3&)>& push_pos,
        const std::function<void(uint32_t, const vec3&)>& push_rot);

    vec3 pos_at(uint32_t id, float t);
    quat rot_at(uint32_t id, float t);

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
    auto& f = info[id];
    for (auto i = f.pos_b + 1, e = f.pos_n + f.pos_b -1; i < e; ++i)
        pos[i].m = algo::vec_grad(pos[i-1].v, pos[i].v, pos[i+1].v, pos[i-1].t, pos[i].t, pos[i+1].t);
    for (auto i = f.rot_b + 1, e = f.rot_n + f.rot_b -1; i < e; ++i)
        rot[i].m = algo::quat_grad(rot[i-1].v, rot[i].v, rot[i+1].v);
}

vec3 Animation::pos_at(uint32_t id, float t) {
    if (t < 0 || t > duration)
        switch (mode) {
            case repeat: t = fmod(t, duration); break;
            case clamp: t = std::clamp(t, 0.f, duration); break;
            default: throw std::runtime_error("Invalid play mode");
        }

    auto& f = info[id];
    auto b = pos.begin() + f.pos_b, e = b + f.pos_n;
    auto i = std::lower_bound(b, e, t, [](const key_vec3& k, float t) {
        return k.t < t;
    });
    if (i == b || i->t - t < 1e-4f) return i->v;
    if (i == e) return (i-1)->v;

    auto& [v0, m0, t0] = *(i-1);
    auto& [v1, m1, t1] = *i;
    t = (t - t0) / (t1 - t0);
    const auto t3 = t*t*t, t2 = t*t;
    const auto h00 = 2*t3 - 3*t2 + 1;
    const auto h01 = 3*t2 - 2*t3;
    const auto h10 = t3 - 2*t2 + t;
    const auto h11 = t3 - t2;

    return v0*h00 + v1*h01 + m0*h10 + m1*h11;
}

quat Animation::rot_at(uint32_t id, float t) {
    if (t < 0 || t > duration)
        switch (mode) {
            case repeat: t = fmod(t, duration); break;
            case clamp: t = std::clamp(t, 0.f, duration); break;
            default: throw std::runtime_error("Invalid play mode");
        }

    auto& f = info[id];
    auto b = rot.begin() + f.rot_b, e = b + f.rot_n;
    auto i = std::lower_bound(b, e, t, [](const key_quat& k, float t) {
        return k.t < t;
    });
    if (i == b || i->t - t < 1e-4f) return i->v;
    if (i == e) return (i-1)->v;

    auto& [v0, m0, t0] = *(i-1);
    auto& [v1, m1, t1] = *i;

    t = (t - t0) / (t1 - t0);
    return slerp(slerp(v0, v1, t), slerp(m0, m1, t), 2*t*(1-t));
}

struct Bone {
    std::unordered_map<uint32_t, Joint> joints;
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
    std::ifstream ifs{file};
    auto json = nlohmann::json::parse(ifs);

    if (json.is_array())
        for (auto&& item : json)
            parse_model(item);
    else parse_model(json);
}

void anim_manager::load_bone(std::string_view file) {
    std::ifstream ifs{file};
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

    auto& cubes = it->second.cubes;
    assert(j["cubes"].is_object());
    for (auto&& [k, v] : j["cubes"].items()) {
        assert(v.is_object());
        auto& c = cubes[s2id(k)];
        if (v.contains("pos")) c.pos = j2vec3(v["pos"]);
        if (v.contains("ext")) c.ext = j2vec3(v["ext"]);
        if (v.contains("cnt")) c.cnt = j2vec3(v["cnt"]);
        if (v.contains("rot")) c.rot = j2quat(v["rot"]);
    }

    if (j.contains("completed") && j["completed"] == false) {
        std::map<uint32_t, uint32_t> p;
        for (auto&& [k, v] : j["cubes"].items()) {
            if (v.contains("parent")) {
                p.emplace(s2id(k), s2id(v["parent"].get<std::string>()));
            }
        }
        for (auto&& [n, m] : p) {
            auto& curr = cubes[n];
            auto& parent = cubes[m];
            auto t = parent.local() * curr.local();
            curr.pos = t[3];
            curr.rot = quat_cast(t);
        }
    }
}

void unfold() {

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

        for (auto&& [k, u] : v.items()) {
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
                std::transform(keys.begin(), keys.end(), std::back_inserter(a), [](auto&& p) {
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
                std::transform(keys.begin(), keys.end(), std::back_inserter(a), [](auto&& p) {
                    return key_vec3{ .v = j2quat(p["v"]), .t = p["t"] };
                });
                std::stable_sort(a.rot.data() + info.rot_b, a.rot.data() + info.rot_b + info.rot_n, [](auto&& a, auto&& b) {
                    return a.t < b.t;
                });
            }
        }
    }

    if (j.contains("completed") && j["completed"] == false) {
        assert(j.contains("joints") && j["joints"].is_object());
        auto& jnts = it->second.joints;
        std::map<uint32_t, uint32_t> p;
        for (auto&& [k, v] : j["joints"].items()) {
            auto [it, succ] = jnts.try_emplace(s2id(k));
            if (!succ) {
                std::cerr << "Duplicate joint: " << k << " in bone: " << j["name"] << std::endl;
                continue;
            }
            auto& jnt = it->second;
            assert(v.is_object());
            if (v.contains("pos")) jnt.pos = j2vec3(v["pos"]);
            if (v.contains("rot")) jnt.rot = j2quat(v["pos"]);
            if (v.contains("parent"))
                p.emplace(s2id(k), s2id(v["parent"].get<std::string>()));
        }
        for (auto&& [n, m] : p) {
            Cube curr = { .pos = jnts[n].pos, .rot = jnts[n].rot };
            Cube parent = { .pos = jnts[m].pos, .rot = jnts[m].rot };
            auto t = parent.local() * curr.local();
            curr.pos = t[3];
            curr.rot = quat_cast(t);

            // TODO
        }
    }

    for (auto&& [n, a] : anims)
        a.calcu_grad();
}

void add_model(const Model& m) {
    for (auto&& [n, c] : m.cubes)
        add_cube(c);
}

int main() {
    init_window();

    anim_manager anim;
    anim.load_model(FILE_ROOT"test.json");

    while (!glfwWindowShouldClose(window)) {
        glClearColor(.2, .3, .3, 1.);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        process_input();

        glUseProgram(sh);
        glUniformMatrix4fv(glGetUniformLocation(sh, "view"), 1, GL_FALSE,
            lookAt(cam, vec3{0}, vec3{0, 1, 0}).value_ptr());

        // anim.models["test"].cubes[s2id("K0")].rot = quat({1, 0, 0}, glfwGetTime());
        add_model(anim.models["test"]);
        draw_cube();
        clear_cube();

        glfwSwapBuffers(window);
        glfwPollEvents();
    }
}