#include "vecto_extension.hpp"
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
    std::vector<f32> out_transform(10);
    std::vector<f32> out_copy(10);
    std::iota(a.begin(), a.end(), 0.f);

    auto a_kwk = kwk::view{kwk::source = a, kwk::of_size(10)};
    auto b_kwk = kwk::view{kwk::source = out_transform, kwk::of_size(10)};
    auto c_kwk = kwk::view{kwk::source = out_copy, kwk::of_size(10)};
    
    std::cout << "a_kwk : " << a_kwk << "\n";
    std::cout << "b_kwk : " << b_kwk << "\n";

    kwk::transform("simd", [](auto e, auto b){return e * 5.f + b ;}, b_kwk, a_kwk, a_kwk);
    std::cout << "Transform : \n"<< b_kwk <<"\n";

    auto res = kwk::reduce("simd", b_kwk);
    std::cout << "Reduction : " << res << "\n";
    
    //auto res_transform = kwk::reduce("simd", b_kwk, [](auto a, auto e){return a-e;} , 2.f);
    //std::cout << res_transform << " reduction result\n";
    auto pred1 = kwk::all_of("simd", a_kwk, [&](auto e){return e < 10;});
    std::cout << "All of : " << pred1 << "\n";

    auto pred2 = kwk::any_of("simd", a_kwk, [&](auto e){return e < 5;});
    std::cout << "Any of : " << pred2 << "\n";

    auto pred3 = kwk::none_of("simd", a_kwk, [&](auto e){return e > 10;});
     std::cout << "None of : " << pred3 << "\n";


    kwk::copy("simd", b_kwk, a_kwk);
    std::cout << "Copy :\n" << b_kwk << "\n";

    kwk::copy_if("simd", [](auto a){return a < 5;}, c_kwk, a_kwk);
    std::cout << "Copy_if <5 :\n" << c_kwk << "\n";

    auto find1 = kwk::find("simd", a_kwk, 5);
    std::cout << "Find             : "<< *find1 << " addr : " << find1 <<"\n";

    auto find2 = kwk::find_if("simd", a_kwk, [](auto a){return a > 4 && a < 6;});
    std::cout << "Find_if          : "<< *find2 << " addr : " << find2 <<"\n";

    auto find3 = kwk::find_if_not("simd", a_kwk, [](auto a){return a <= 4 ;});
    std::cout << "Find_if_not      : "<< *find3 << " addr : " << find3 <<"\n";

    auto find4 = kwk::find_last("simd", a_kwk, 7);
    std::cout << "Find_last        : "<< *find4 << " addr : " << find4 <<"\n";
    
    auto find5 = kwk::find_last_if("simd", a_kwk, [](auto a){return a > 6 && a < 8;});
    std::cout << "Find_last_if     : "<< *find5 << " addr : " << find5 <<"\n";

    auto find6 = kwk::find_last_if_not("simd", a_kwk, [](auto a){return a > 7;});
    std::cout << "Find_last_if_not : "<< *find6 << " addr : " << find6 <<"\n";

    auto find7 = kwk::find_first_of("simd", a_kwk, b_kwk);
    std::cout << "Find_first_of    : "<< find7 <<"\n";

    return 0;
}
