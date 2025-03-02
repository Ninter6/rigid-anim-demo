//
// Created by Ninter6 on 2025/3/1.
//

#include <iostream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>

#include "stb/stb_image.h"

#include "json.hpp"

constexpr size_t PKG_SIZE = 4096;

struct Image {
    int w,  h;
    uint32_t* data;
};

std::vector<Image> imgs;
uint32_t pkg[PKG_SIZE][PKG_SIZE]{};

struct Rect { int x, y, w, h; };
struct Unit {
    Rect rect;
    size_t img;
    std::string arr;
};

std::vector<Unit> units;

bool rect_cover(const Rect& a, const Rect& b) {
    return a.x + a.w >= b.x && a.y + a.h >= b.y && a.x <= b.x + b.w && a.y <= b.y + b.h;
}

void load_img(std::string_view file) {
    int channels;
    auto& [w, h, data] = imgs.emplace_back();
    data = (uint32_t*)stbi_load(file.data(), &w, &h, &channels, 4);
    if (!data) {
        std::cerr << "Failed to load image: " << stbi_failure_reason() << '\n';
        imgs.pop_back();
    }
}

void load(const nlohmann::json& j) {
    assert(j.is_object() && j.contains("images") && j.contains("arrays"));

    const auto& images = j["images"];
    assert(images.is_array());
    for (auto&& i : images)
        load_img(i.get<std::string>());

    const auto& arrays = j["arrays"];
    assert(arrays.is_object());
    std::unordered_map<size_t, std::vector<size_t>> img_unit;
    for (auto&& [k, v] : arrays.items()) {
        assert(v.is_array());
        for (int i = 0; i < v.size(); ++i) {
            const auto& u = v[i];
            auto add_unit = [&k, &img_unit](auto&& u, auto&& n) {
                assert(u.is_object());
                auto&& [rect, img, id] = units.emplace_back();
                img = u["img"];
                rect.x = u["uv"][0];
                rect.y = u["uv"][1];
                rect.w = u["uv"][2];
                rect.h = u["uv"][3];
                id = k + ':';
                id += n;
                img_unit[img].push_back(units.size()-1);
            };
            auto ni = std::to_string(i);
            if (u.is_array()) {
                ni += ':';
                for (int j = 0; j < u.size(); ++j)
                    add_unit(u[j], ni + std::to_string(j));
            } else add_unit(u, ni);
        }
    }

    for (auto&& [img, us] : img_unit) {
        std::sort(us.begin(), us.end(), [](auto&& a, auto&& b) {
            auto&& ra = units[a].rect;
            auto&& rb = units[b].rect;
            return ra.w*ra.h > rb.w*rb.h;
        });
        std::vector<Rect> rs;
        std::vector<std::vector<int>> rect_unit;
        rs.resize(us.size());
        rect_unit.resize(us.size());
        std::transform(us.begin(), us.end(), rs.begin(), [](auto&&i) {
            return units[i].rect;
        });
        for (int i = 0; i < us.size(); ++i)
            rect_unit[i].push_back(i);
        // std::unordered_set<int> rm;
        for (int i = 0; i < rs.size(); ++i)
            if (!rect_unit[i].empty())
                for (int j = i+1; j < rs.size(); ++j)
                    if (!rect_unit[j].empty() && rect_cover(rs[i], rs[j])) {
                        std::copy(rect_unit[j].begin(), rect_unit[j].end(), std::back_inserter(rect_unit[i]));
                        rect_unit[j].clear();
                    }
        for (int i = 0; auto&& u : rect_unit)
            if (u.empty()) rs.erase(rs.begin() + i);
            else ++i;
        std::erase_if(rect_unit, [](auto&& u) {return u.empty();});
    }
}

int main() {
    std::cout << "Hello world!\n";
}
