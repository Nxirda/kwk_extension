#include "transform_vectorized.hpp"
#include <iostream>
#include <vector>
#include <numeric>

using f32 = float;

int main(int argc, char **argv)
{
    if(argc == 0)
    {
        std::cout <<"Usage is " << argv[0] << "\n";
    }

    std::vector<f32> a(10);
    std::vector<f32> out_kwk(10);
    std::vector<f32> out_eve(10);
    std::iota(a.begin(), a.end(), 0.f);

    auto a_kwk = kwk::view{kwk::source = a, kwk::of_size(10)};
    auto b_kwk = kwk::view{kwk::source = out_kwk, kwk::of_size(10)};
    
    eve::algo::transform_to(a, out_eve, [](auto e){return e+5;});
    std::cout << kwk::view{kwk::source = out_eve, kwk::of_size(10)};

    kwk::transform("simd", [](auto e){return e + 5;}, b_kwk, a_kwk);
    std::cout << b_kwk << "\n";
      
    return 0;
}
