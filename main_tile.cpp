#include "vecto_extension.hpp"
#include "tile_imp.hpp"

#include <chrono>
#include <iostream>
#include <vector>
#include <numeric>

using f32 = float;

// Basic main just to verify tiling
int main()
{
    std::vector<f32> b(16);
    std::iota(b.begin(), b.end(), 0.f);
    
    auto base_shape = kwk::of_size(4,4);
    auto base = kwk::table{kwk::source = b, base_shape};

    const auto tile_shape = kwk::of_size(2,2);

    const auto t1 = overlapping_tiles(base, tile_shape);

    const auto t2 = paving_tiles(base, tile_shape, kumi::make_tuple(1,1));

    const auto t3 = paving_tiles(base, tile_shape, kumi::make_tuple(1,3));
    
    const auto t4 = paving_tiles(base, tile_shape, kumi::make_tuple(3, 1));
    std::cout << base << "\n";
    std::cout << t1 << "\n";
    std::cout << t2 << "\n";
    std::cout << t3 << "\n";
    std::cout << t4 << "\n";

    return 0;
}
