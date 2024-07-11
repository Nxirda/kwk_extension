cmake_minimum_required(VERSION 3.18)
enable_testing()

project(kwk_extension LANGUAGES CXX DESCRIPTION "Kwk extensions dev and tests")

set(CMAKE_COLOR_DIAGNOSTICS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)

add_compile_options(-Wall -Wconversion -O3 -march=native -g)

find_package(eve CONFIG REQUIRED)
find_package(kiwaku CONFIG REQUIRED)
find_package(kumi CONFIG REQUIRED)

add_executable(main main.cpp)
target_link_libraries(main PRIVATE kiwaku::kiwaku eve::eve kumi::kumi)

add_executable(tile main_tile.cpp)
target_link_libraries(tile PRIVATE kiwaku::kiwaku eve::eve kumi::kumi)

include(CTest)
add_custom_target(unit)
add_subdirectory(test)
add_subdirectory("${PROJECT_SOURCE_DIR}/test/integration")
