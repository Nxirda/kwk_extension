cmake_minimum_required(VERSION 3.16)
project(kwk_extension LANGUAGES CXX DESCRIPTION "Kwk extensions dev and tests")

set(CMAKE_COLOR_DIAGNOSTICS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_options(-Wall -O3 -march=native -g)

find_package(eve CONFIG REQUIRED)
find_package(kiwaku CONFIG REQUIRED)
find_package(kumi CONFIG REQUIRED)

add_executable(main main.cpp)
target_link_libraries(main PRIVATE kiwaku::kiwaku eve::eve kumi::kumi)
