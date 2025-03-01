#include <iostream>
#include <fstream>
#include <sstream>
#include <memory>
#include <chrono>
#include <vector>
#include <stack>
#include <span>
#include <map>
#include <cassert>

#include "glad/glad.h"
#include "GLFW/glfw3.h"

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

struct Cube {
    vec3 pos{}; // left bottom front
    vec3 ext{1}; // extent of the cube
    vec3 cnt{}; // center of rotation
    quat rot{}; // rotation

    mat4 matrix() const {
        auto tr = rotate(rot);
        tr[3] = vec4{pos + cnt, 1.f};
        auto ts = mat4::zero();
        ts[0][0] = ext.x;
        ts[1][1] = ext.y;
        ts[2][2] = ext.z;
        ts[3] = vec4{-cnt, 1.f};
        return tr * ts;
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

struct BasicTimeAxis {
    virtual ~BasicTimeAxis() = default;
    [[nodiscard]] virtual BasicTimeAxis* clone() const = 0;
    [[nodiscard]] virtual const std::type_info& type_info() const = 0;

    virtual void tick(float t, void* p) = 0;
    virtual void calcu_grad()  = 0;

    virtual void tick_and_lerp(float t, void* p, BasicTimeAxis* other, float lt) = 0;
};

template <class T>
struct TimeAxisImpl : BasicTimeAxis {
    [[nodiscard]] BasicTimeAxis* clone() const override {
        return new T(static_cast<T>(*this));
    }
};

template <template <class> class TL, class T>
struct TimeAxisImpl<TL<T>> : BasicTimeAxis {
    using type = TL<T>;
    [[nodiscard]] BasicTimeAxis* clone() const override {
        return new type(static_cast<const type&>(*this));
    }
    [[nodiscard]] const std::type_info& type_info() const override {
        return typeid(T);
    }
    void tick_and_lerp(float t, void* ptr, BasicTimeAxis* o, float lt) override {
        assert(type_info() == o->type_info());
        auto p = static_cast<T*>(ptr);
        T tmp0{}, tmp1{};
        this->tick(t, &tmp0);
        o->tick(t, &tmp1);
        *p = tmp0*(1-lt) + tmp1*lt;
    }
    void calcu_grad() override {}
    T* p = nullptr;
};

template <template <class> class TL, class U>
struct TimeAxisImpl<TL<qua<U>>> : BasicTimeAxis {
    using T = qua<U>;
    using type = TL<T>;
    [[nodiscard]] BasicTimeAxis* clone() const override {
        return new type(static_cast<const type&>(*this));
    }
    [[nodiscard]] const std::type_info& type_info() const override {
        return typeid(T);
    }
    void tick_and_lerp(float t, void* ptr, BasicTimeAxis* o, float lt) override {
        assert(type_info() == o->type_info());
        auto p = static_cast<T*>(ptr);
        T tmp0{}, tmp1{};
        this->tick(t, &tmp0);
        o->tick(t, &tmp1);
        *p = slerp(tmp0, tmp1, lt);
    }
    void calcu_grad() override {}
    T* p = nullptr;
};

template <class T>
struct SmoothTimeAxis : TimeAxisImpl<SmoothTimeAxis<T>> {
    SmoothTimeAxis() = default;
    SmoothTimeAxis(float dur) : duration(dur) {}
    void tick(float t, void* ptr) override {
        auto p = static_cast<T*>(ptr);
        assert(p && duration > 1e-4f);
        t = std::fmod(t, duration);
        auto it = keyframes.lower_bound(t);
        if (it == keyframes.end()) { *p = (--it)->second.first; return; }
        if (it == keyframes.begin() || it->first - t < 1e-4f) {
            *p = it->second.first; return;
        }

        auto m2 = ++it == keyframes.end() ?
            T(0) : it->second.second;
        const auto& [t1, r1] =*--it;
        const auto& [y1, m1] = r1;
        const auto& [t0, r0] =*--it;
        const auto& [y0, m0] = r0;

        t = (t - t0) / (t1 - t0);
        const auto t3 = t*t*t, t2 = t*t;
        const auto n0 = (m0 + m1) * .5f;
        const auto n1 = (m1 + m2) * .5f;
        const auto h00 = 2*t3 - 3*t2 + 1;
        const auto h01 = 3*t2 - 2*t3;
        const auto h10 = t3 - 2*t2 + t;
        const auto h11 = t3 - t2;
        *p = y0*h00 + y1*h01 + n0*h10 + n1*h11;
    }
    void calcu_grad() override {
        if (keyframes.size() < 2) return;
        auto b = keyframes.begin();
        b->second.second = T(0);
        for (auto it = ++b; it != keyframes.end(); ++it) {
            auto it0 = it;
            const auto& [t0, r0] =*--it0;
            const auto& [y0, m0] = r0;
            auto& [t, r] =*it;
            auto& [y, m] = r;
            m = (y - y0) / (t - t0);
        }
    }
    void add_keyframe(float t, const T& v) {
        keyframes[t] = {v, T(0)};
    }
    std::map<float, std::pair<T, T>> keyframes;
    float duration = 0;
};

template <class U>
struct SmoothTimeAxis<qua<U>> : TimeAxisImpl<SmoothTimeAxis<qua<U>>> {
    SmoothTimeAxis() = default;
    SmoothTimeAxis(float dur) : duration(dur) {}
    using T = qua<U>;
    static constexpr auto ln(const T& q) {
        auto& [w, x, y, z] = q.asArray;
        return vec<U, 3>(x, y, z)*(acos(w)/sqrt(1-w*w));
    }
    static constexpr T exp(const vec<U, 3>& v) {
        auto l = v.length();
        if (l < 1e-4f) return T{};
        return {cos(l), v*(sin(l)/l)};
    }
    void tick(float t, void* ptr) override {
        auto p = static_cast<T*>(ptr);
        assert(p && duration > 1e-4f);
        t = std::fmod(t, duration);
        auto it = keyframes.lower_bound(t);
        if (it == keyframes.end()) { *p = (--it)->second.first; return; }
        if (it == keyframes.begin() || it->first - t < 1e-4f) {
            *p = it->second.first; return;
        }

        const auto& [t1, r1] =*it;
        const auto& [q1, s1] = r1;
        const auto& [t0, r0] =*--it;
        const auto& [q0, s0] = r0;

        t = (t - t0) / (t1 - t0);
        *p = slerp(slerp(q0, q1, t), slerp(s0, s1, t), 2*t*(1-t));
    }
    void calcu_grad() override {
        if (keyframes.size() < 3) return;
        auto b = keyframes.begin();
        auto e = --keyframes.end();
        for (auto it = ++b; it != e;) {
            const auto& [t0, r0] = *--it;
            const auto& [q0, s0] = r0;
                  auto& [t1, r1] = *++it;
                  auto& [q1, s1] = r1;
            const auto& [t2, r2] = *++it;
            const auto& [q2, s2] = r2;
            auto q1s = q1.conjugate();
            auto lnq1sq0 = ln(q1s*q0);
            auto lnq1sq2 = ln(q1s*q2);
            s1 *= exp((lnq1sq0+lnq1sq2)*U(-.25));
        }
    }
    void add_keyframe(float t, const T& q) {
        keyframes[t] = {q, q};
    }
    std::map<float, std::pair<T, T>> keyframes;
    float duration = 0;
};

template <class T>
struct FlatTimeAxis : TimeAxisImpl<FlatTimeAxis<T>> {
    FlatTimeAxis() = default;
    FlatTimeAxis(float dur) : duration(dur) {}
    void tick(float t, void* ptr) override {
        auto p = static_cast<T*>(ptr);
        assert(p);
        t = std::fmod(t, duration);
        auto it = keyframes.lower_bound(t);
        if (it == keyframes.end()) { *p = (--it)->second; return; }
        if (it == keyframes.begin() || it->first - t < 1e-4f) {
            *p = it->second; return;
        }

        const auto& [t1, y1] = *it;
        const auto& [t0, y0] = *--it;
        t = (t - t0) / (t1 - t0);
        *p = lerp(y0, y1, t);
    }
    void add_keyframe(float t, const T& v) {
        keyframes[t] = v;
    }
    std::map<float, T> keyframes;
    float duration = 0;
};

template <class U>
struct FlatTimeAxis<qua<U>> : TimeAxisImpl<FlatTimeAxis<qua<U>>> {
    FlatTimeAxis() = default;
    FlatTimeAxis(float dur) : duration(dur) {}
    using T = qua<U>;
    void tick(float t, void* ptr) override {
        auto p = static_cast<T*>(ptr);
        assert(p);
        t = std::fmod(t, duration);
        auto it = keyframes.lower_bound(t);
        if (it == keyframes.end()) { *p = (--it)->second; return; }
        if (it == keyframes.begin() || it->first - t < 1e-4f) {
            *p = it->second; return;
        }

        const auto& [t1, y1] = *it;
        const auto& [t0, y0] = *--it;
        t = (t - t0) / (t1 - t0);
        *p = slerp(y0, y1, t);
    }
    void add_keyframe(float t, const T& v) {
        keyframes[t] = v;
    }
    std::map<float, T> keyframes;
    float duration = 0;
};

struct Animation {
    Animation() = default;
    // template <template <class> class T, class U, class =
    //     std::enable_if_t<std::is_base_of_v<BasicTimeAxis, T<U>>>>
    // T<U>& add_time_axis(U* p_to_bind) {
    //     auto* axis = new T<U>;
    //     axis->bind(p_to_bind);
    //     axes.emplace(p_to_bind, axis);
    //     return *axis;
    // }
    template <class T>
    void add_time_axis(BasicTimeAxis* axis, T* p) {
        if constexpr (!std::is_same_v<T, void>)
            assert(typeid(T) == axis->type_info());
        axes.emplace(p, axis);
    }
    void tick(float t) {
        for (auto&& [p, axis] : axes)
            axis->tick(t, p);
    }
    std::map<void*, BasicTimeAxis*> axes;
};

struct CubeModel {
    static constexpr auto null_index = UINT32_MAX;
    struct Node {
        Cube cube{};
        uint32_t gen = 0;
        uint32_t parent = null_index;
        uint32_t brother = null_index;
    };

    CubeModel() = default;
    static CubeModel load_from_file(std::string_view file) {
        CubeModel m;
        std::ifstream ifs(file);
        if (!ifs.good())
            throw std::runtime_error{"Failed to open model file"};

        int gen = 0; // generation
        while (!ifs.eof()) {
            std::string buf;
            std::stringstream ss;
            do {do { std::getline(ifs, buf); buf = buf.substr(0, buf.find("//")); }
                while (buf.find_first_not_of(" \r\n") == std::string::npos);
                ss << buf << ' ';
            } while (ifs.get() == '-');
            if (!ifs.eof()) ifs.seekg(-1, std::ios::cur);

            Node node;
            auto& g = node.gen;
            while (ss.get() == '>') ++g;
            ss.seekg(-1, std::ios::cur);
            if (g < gen && g != 0) {
                node.parent = m.nodes.back().parent;
                while (g < gen--) {
                    node.brother = node.parent;
                    node.parent = m.nodes[node.parent].parent;
                }
                assert(m.nodes[node.brother].brother == null_index);
                std::swap(m.nodes[node.brother].brother, node.brother);
            } else if (g > gen) {
                assert(g == gen + 1);
                ++gen;
                node.parent = m.nodes.size() - 1;
            } else if (!m.nodes.empty()) {
                auto id = m.nodes.size();
                auto& last = m.nodes.back();
                last.brother = id;
                node.parent = last.parent;
            }
            auto& c = node.cube;
            ss >> c.pos.x >> c.pos.y >> c.pos.z
               >> c.ext.x >> c.ext.y >> c.ext.z
               >> c.cnt.x >> c.cnt.y >> c.cnt.z
               >> c.rot.w >> c.rot.x >> c.rot.y >> c.rot.z;
            m.nodes.push_back(std::move(node));
        }

        return m;
    }

    std::vector<mat4> matrices(const mat4& ori = {}) const {
        std::vector<mat4> ms;
        std::stack<mat4> st;
        st.push(ori);
        for (const auto& n : nodes) {
            while (st.size() - 1 > n.gen)
                st.pop();
            auto tr = rotate(n.cube.rot);
            tr[3] = vec4{n.cube.pos + n.cube.cnt, 1.f};
            auto tc = translate(-n.cube.cnt);
            auto ts = mat4::zero();
            ts[0][0] = n.cube.ext.x;
            ts[1][1] = n.cube.ext.y;
            ts[2][2] = n.cube.ext.z;
            ts[3] = tc[3];
            tr = st.top() * tr;
            ms.push_back(tr * ts);
            st.push(tr * tc);
        }
        return ms;
    }

    std::vector<Node> nodes;
};

void add_model(const CubeModel& m) {
    auto mat = m.matrices();
    std::memcpy(m_map + m_count, mat.data(), mat.size() * sizeof(mat4));
    m_count += mat.size();
}

std::vector<std::pair<size_t, BasicTimeAxis*>>
load_animation(std::string_view file) {
    std::ifstream ifs{file};
    if (!ifs.good())
        throw std::runtime_error{"Failed to open animation file"};
    std::vector<vec3> ps;
    std::vector<vec3> es;
    std::vector<vec3> cs;
    std::vector<quat> rs;
    std::vector<std::tuple<BasicTimeAxis*, std::string, size_t>> ls;

    std::string buf;
    bool flag = false;
    while (true) {
        do {std::getline(ifs, buf);
            auto n = buf.find_first_not_of(" \r\n");
            if (n == std::string::npos || buf[n] == '/') {
                buf.clear();
                continue;
            }
            if (!std::isalpha(buf[n])) {
                ifs.seekg(-buf.size()-1, std::ios::cur);
                flag = true;
                break;
            }
            buf = buf.substr(n, buf.find("//") - n);
        } while (buf.empty() && !ifs.eof());
        if (flag || buf.empty()) break;

        std::stringstream ss;
        ss << buf;
        ss >> buf;
        switch (buf[0]) {
            case 'p': {
                auto& p = ps.emplace_back();
                ss >> p.x >> p.y >> p.z;
                break;
            }
            case 'e': {
                auto& e = es.emplace_back();
                ss >> e.x >> e.y >> e.z;
                break;
            }
            case 'c': {
                auto& c = cs.emplace_back();
                ss >> c.x >> c.y >> c.z;
                break;
            }
            case 'r': {
                auto& r = rs.emplace_back();
                ss >> r.w >> r.x >> r.y >> r.z;
                break;
            }
            case 'f': {
                float dur;
                auto& [l, s, id] = ls.emplace_back();
                ss >> buf >> id >> dur;
                s = 'f' + buf.erase(1);
                id = (id << 8) | buf[0];
                switch (buf[0]) {
                    case 'p':
                    case 'e':
                    case 'c': l = new FlatTimeAxis<vec3>(dur); break;
                    case 'r': l = new FlatTimeAxis<quat>(dur); break;
                    default: assert(false);
                }
                break;
            }
            case 's': {
                float dur;
                auto& [l, s, id] = ls.emplace_back();
                ss >> buf >> id >> dur;
                s = 's' + buf.erase(1);
                id = (id << 8) | buf[0];
                switch (buf[0]) {
                    case 'p':
                    case 'e':
                    case 'c': l = new SmoothTimeAxis<vec3>(dur); break;
                    case 'r': l = new SmoothTimeAxis<quat>(dur); break;
                    default: assert(false);
                }
                break;
            }
            default: assert(false);
        }
    }

    while (std::getline(ifs, buf)) {
        std::stringstream ss;
        int aid, vid;
        float time;
        ss << buf.substr(0, buf.find("//"));
        if (buf.find_first_not_of(" \r\n") == std::string::npos)
            continue;
        ss >> aid >> vid >> time;
        const auto& [tx, at, id] = ls[aid];
        switch (at[1]) {
            case 'p':
                if (at[0] == 'f')
                    static_cast<FlatTimeAxis<vec3>*>(tx)->add_keyframe(time, ps[vid]);
                else
                    static_cast<SmoothTimeAxis<vec3>*>(tx)->add_keyframe(time, ps[vid]);
                break;
            case 'e':
                if (at[0] == 'f')
                    static_cast<FlatTimeAxis<vec3>*>(tx)->add_keyframe(time, es[vid]);
                else
                    static_cast<SmoothTimeAxis<vec3>*>(tx)->add_keyframe(time, es[vid]);
                break;
            case 'c':
                if (at[0] == 'f')
                    static_cast<FlatTimeAxis<vec3>*>(tx)->add_keyframe(time, cs[vid]);
                else
                    static_cast<SmoothTimeAxis<vec3>*>(tx)->add_keyframe(time, cs[vid]);
                break;
            case 'r':
                if (at[0] == 'f')
                    static_cast<FlatTimeAxis<quat>*>(tx)->add_keyframe(time, rs[vid]);
                else
                    static_cast<SmoothTimeAxis<quat>*>(tx)->add_keyframe(time, rs[vid]);
                break;
            default: assert(false);
        }
    }

    std::vector<std::pair<size_t, BasicTimeAxis*>> rst;
    for (auto&& [tx, at, id] : ls) {
        if (at[0] == 's') tx->calcu_grad();
        rst.emplace_back(id, tx);
    }

    return rst;
}

Animation load_and_bind_animation(std::string_view file, CubeModel& model) {
    auto l = load_animation(file);

    Animation anim;
    for (auto&& [id, tx] : l) {
        auto t = id & 0xff;
        assert((id >> 8) < model.nodes.size());
        auto& cube = model.nodes[id >> 8].cube;
        switch (t) {
            case 'p': anim.add_time_axis(tx, &cube.pos); break;
            case 'e': anim.add_time_axis(tx, &cube.ext); break;
            case 'c': anim.add_time_axis(tx, &cube.cnt); break;
            case 'r': anim.add_time_axis(tx, &cube.rot); break;
            default: assert(false);
        }
    }

    return anim;
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

int main() {
    std::cout << "Hello, World!" << std::endl;

    init_window();

    auto model = CubeModel::load_from_file(FILE_ROOT"test.txt");

    // std::vector<Cube> cubes;
    // cubes.emplace_back() = {.ext = {2, 2, .5f}, .cnt = {1.f, 1.f, .25f}};
    // cubes.push_back(model.nodes[0].cube);

    Animation anim = load_and_bind_animation(FILE_ROOT"test_run.txt", model);

    // auto& tl0 =
    //     anim.add_time_axis<SmoothTimeAxis>(&cubes.back().pos);
    // tl0.duration = 4;
    // tl0.add_keyframe(-.9f, {-1.f, 1.f, -.25f});
    // tl0.add_keyframe(1, {-1.f, -3.f, -.25f});
    // tl0.add_keyframe(3, {-1.f, 1.f, -.25f});
    // tl0.add_keyframe(4.9f, {-1.f, -3.f, -.25f});
    // tl0.calcu_grad();
    //
    // auto& tl1 =
    //     anim.add_time_axis<SmoothTimeAxis>(&cubes.back().rot);
    // tl1.duration = 4;
    // tl1.add_keyframe(0, quat{{0, 1, 0}, M_PI * 0.f});
    // tl1.add_keyframe(1, quat{{1, 0, 0}, M_PI * .5f});
    // tl1.add_keyframe(2, quat{{0, 1, 0}, M_PI * 1.f});
    // tl1.add_keyframe(3, quat{{0, 0, 1}, M_PI *1.5f});
    // tl1.add_keyframe(4, quat{{0, 1, 0}, M_PI * 2.f});
    // tl1.calcu_grad();

    using cl = std::chrono::system_clock;
    auto start = cl::now();
    while (!glfwWindowShouldClose(window)) {
        glClearColor(0.2, 0.3, 0.3, 1.0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        process_input();
        auto t = std::chrono::duration<float>(cl::now() - start).count();
        anim.tick(t);

        glUseProgram(sh);
        glUniformMatrix4fv(glGetUniformLocation(sh, "view"), 1, GL_FALSE,
            lookAt(cam, vec3{0}, vec3{0, 1, 0}).value_ptr());

        // add_cube(model.nodes[0].cube);
        add_model(model);
        draw_cube();
        clear_cube();

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    return 0;
}