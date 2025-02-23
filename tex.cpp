//
// Created by Ninter6 on 2025/2/5.
//

#include "stb_image.h"
#include "stb_image_resize2.h"
#include "stb_image_write.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cassert>

#include "mathpls.h"

using namespace mathpls;

constexpr ivec2 rect_size = {32};
constexpr ivec2 tex_size = {6, 12};
constexpr ivec2 final_size = {rect_size.x * tex_size.x, rect_size.y * tex_size.y};

std::string tex_name = "no_title";
std::vector<ivec3> exts;
std::vector<ivec2> sizes;
std::vector<uint8_t*> datas;
auto final = new uint8_t[final_size.x * final_size.y << 2]{};

uint8_t* load_image(std::string_view file, ivec2& size) {
    int channels;
    auto data = stbi_load(file.data(), &size.x, &size.y, &channels, 4);
    if (!data) {
        std::cerr << "Failed to load image: " << stbi_failure_reason() << ' ' << file << '\n';
        std::terminate();
    }
    std::cout << "Image loaded: " << file << ", size: " << size.x << "x" << size.y << ", channels: " << channels << '\n';
    return data;
}

void load_config(std::string_view config) {
    std::ifstream ifs{config};
    if (!ifs.good())
        throw std::runtime_error("Failed to open:" + std::string(config));
    auto path = std::string(config.substr(0, config.rfind('/')+1));

    for (std::string buf; ifs >> buf;) {
        if (buf == "n") {
            ifs >> tex_name;
            continue;
        }
        size_t id = 0;
        if (std::isdigit(buf[0]))
            id = std::stoi(buf);
        else switch (buf[0]) {
            case 'b': id = 0; break;
            case 'h': id = 1; break;
            case 'a': id = buf[1] == 'l' ? 2 : 3; break;
            case 'l': id = buf[1] == 'l' ? 4 : 5; break;
            default: assert(false);
        }
        assert(id < 12);
        ivec3 ext;
        ifs >> buf;
        ifs >> ext.x >> ext.y >> ext.z;
        exts.resize(id + 1);
        sizes.resize(id + 1);
        datas.resize(id + 1);
        exts[id] = ext;
        datas[id] = load_image(path + buf, sizes[id]);
    }
}

void image_resize(uint8_t* src, ivec2 sz, uint8_t* dst) {
    auto rst = stbir_resize_uint8_linear(
        src, sz.x, sz.y, 0,
        dst, rect_size.x, rect_size.y, 0,
        STBIR_RGBA);
    if (!rst) {
        std::cerr << "Failed to resize image: " << stbi_failure_reason() << '\n';
        std::terminate();
    }
}

void process_image() {
    for (int i = 0; i < datas.size(); ++i) {
        auto data = datas[i];
        auto size = sizes[i];
        auto ext = exts[i];
        if (!data) continue;

        assert((ext.x + ext.z) * 2 == size.x);
        assert(ext.y + ext.z * 2 == size.y);

        auto front = new uint8_t[ext.x+ext.y<<2];
        auto left = new uint8_t[ext.z+ext.y<<2];
        auto back = new uint8_t[ext.x+ext.y<<2];
        auto right = new uint8_t[ext.z+ext.y<<2];
        auto top = new uint8_t[ext.x+ext.z<<2];
        auto bottom = new uint8_t[ext.x+ext.z<<2];
        auto buf = new uint8_t[6][rect_size.x*rect_size.y<<2]{};

        int tl = size.x << 2;
        int o = ext.z * tl, ll = ext.x << 2, nl = ext.y;

        for (int j = 0; j < nl; ++j)
            std::memcpy(front + ll*j, data + o + tl*j, ll);
        image_resize(front, {ext.x, ext.y}, buf[0]);
        o = ext.z * tl + (ext.x<<2); ll = ext.z << 2;
        for (int j = 0; j < nl; ++j)
            std::memcpy(left + ll*j, data + o + tl*j, ll);
        image_resize(left, {ext.z, ext.y}, buf[1]);
        o = ext.z * tl + (ext.x + ext.z << 2); ll = ext.x << 2;
        for (int j = 0; j < nl; ++j)
            std::memcpy(back + ll*j, data + o + tl*j, ll);
        image_resize(back, {ext.x, ext.y}, buf[2]);
        o = ext.z * tl + (ext.x<<3) + (ext.z<<2); ll = ext.z << 2;
        for (int j = 0; j < nl; ++j)
            std::memcpy(right + ll*j, data + o + tl*j, ll);
        image_resize(right, {ext.z, ext.y}, buf[3]);
        o = 0; nl = ext.z; ll = ext.x << 2;
        for (int j = 0; j < nl; ++j)
            std::memcpy(top + ll*j, data + o + tl*j, ll);
        image_resize(top, {ext.x, ext.z}, buf[4]);
        o = (ext.z + ext.y) * tl;
        for (int j = 0; j < nl; ++j)
            std::memcpy(bottom + ll*j, data + o + tl*j, ll);
        image_resize(bottom, {ext.x, ext.z}, buf[5]);

        for (int j = 0; j < 6; ++j) {
            for (int k = 0; k < rect_size.y; ++k)
                std::memcpy(
                    final + (k + i*rect_size.y) * final_size.x*4 + j*rect_size.x*4,
                    buf[j] + k * (rect_size.x<<2),
                    rect_size.x<<2);
        }
    }
}

int main(int argc, char** argv) {
    std::cout << "Hello world!\n";

    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <config>\n";
        return 1;
    }

    std::string config = argv[1];

    load_config(config);
    process_image();

    std::string output = config.substr(0, config.rfind('/')+1)+tex_name+".png";
    stbi_write_png(output.c_str(), final_size.x, final_size.y, 4, final, 0);
    std::cout << "Image saved: " << output << '\n';

    for (auto& data : datas)
        stbi_image_free(data);
    delete[] final;
}