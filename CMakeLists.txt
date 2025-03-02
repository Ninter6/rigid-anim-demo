cmake_minimum_required(VERSION 3.20)
project(rigid_anim)

set(CMAKE_CXX_STANDARD 20)

add_compile_definitions(-DFILE_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/")

find_package(GLFW3 REQUIRED)

add_executable(rigid_anim main.cpp glad/glad.c)
target_link_libraries(rigid_anim glfw)
target_compile_options(rigid_anim PUBLIC -O3 -march=native)

add_executable(rigid_anim1 main1.cpp glad/glad.c)
target_link_libraries(rigid_anim1 glfw)
#target_compile_options(rigid_anim1 PUBLIC -O3 -march=native)

file(GLOB STB stb/*)
set_source_files_properties(${STB} PROPERTIES LANGUAGE C)
add_library(stb ${STB})
target_include_directories(stb PUBLIC stb/)
target_compile_definitions(stb PRIVATE
        "STB_IMAGE_IMPLEMENTATION"
        "STB_IMAGE_RESIZE_IMPLEMENTATION"
        "STB_IMAGE_WRITE_IMPLEMENTATION")
target_compile_options(stb PRIVATE -O3 -march=native -w)

add_executable(tex tex.cpp)
target_link_libraries(tex stb)

add_executable(img_packer img_packer.cpp)
target_link_libraries(img_packer stb)
