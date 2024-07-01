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
    std::iota(a.begin(), a.end(), 3.f);

    auto a_kwk = kwk::view{kwk::source = a, kwk::of_size(2,5)};
    auto b_kwk = kwk::view{kwk::source = out_transform, kwk::of_size(2,5)};
    auto c_kwk = kwk::view{kwk::source = out_copy, kwk::of_size(2,5)};
    
    std::cout << "a_kwk : " << a_kwk << "\n";
    std::cout << "b_kwk : " << b_kwk << "\n";

    kwk::transform("simd", [](auto e, auto b){return e * 5.f + b ;}, b_kwk, a_kwk, a_kwk);
    kwk::transform([](auto e, auto b){return e * 5.f + b;}, c_kwk, a_kwk, a_kwk);
    std::cout << "Transform     : \n" << b_kwk  << c_kwk << "\n";


    auto res = kwk::reduce("simd", b_kwk);
    auto res2 = kwk::reduce(c_kwk);
    std::cout << "Reduction         : " << res << " | "<< res2 << "\n";

   
    auto res_reduce = kwk::reduce("simd", b_kwk, std::pair{eve::add, 0}, 2.f);
    auto res_reduce2 = kwk::reduce(c_kwk, [](auto a, auto b){return a+b;}, 2.f);
    std::cout << "Reduction with op : " << res_reduce <<  " | " << res_reduce2 << "\n";

    auto res_trs = kwk::transform_reduce("simd", b_kwk, b_kwk, 0.0, std::pair{eve::add, 0}, [](auto a, auto b){return a+b;} );
    auto res_trs2 = kwk::transform_reduce(b_kwk, b_kwk, 0.0, [](auto a, auto b){return a+b;}, [](auto a, auto b){return a+b;} );
    std::cout << "Transform_reduce  : " << res_trs << " | " << res_trs2 << "\n";


    kwk::transform_inclusive_scan(a_kwk, b_kwk, 1.f, [](auto a, auto b){return a+b;}, [](auto e){return e;});
    kwk::transform_inclusive_scan("simd", a_kwk, c_kwk, 1.f, std::pair{eve::add, 0} , [](auto e){return e;});
    std::cout << "Inclusive scan    : \n" << b_kwk  << c_kwk << "\n";


    kwk::transform_exclusive_scan(a_kwk, b_kwk, 1.f, [](auto a, auto b){return a+b;}, [](auto e){return e;});
    kwk::transform_exclusive_scan("simd", a_kwk, c_kwk, 1.f, std::pair{eve::add, 0}, [](auto e){return e;});
    std::cout << "Exclusive scan    : \n" << b_kwk << c_kwk << "\n";


    auto pred1 = kwk::all_of("simd", a_kwk, [&](auto e){return e > 2;});
    auto pred1_2 = kwk::all_of(a_kwk, [&](auto e){return e > 2;});
    std::cout << "All of             : " << pred1 << " | " << pred1_2 << "\n";

    auto pred2 = kwk::any_of("simd", a_kwk, [&](auto e){return e < 5;});
    auto pred2_2 = kwk::any_of(a_kwk, [&](auto e){return e < 5;});
    std::cout << "Any of             : " << pred2 << " | " << pred2_2 << "\n";

    auto pred3 = kwk::none_of("simd", a_kwk, [&](auto e){return e < 1;});
    auto pred3_2 = kwk::none_of(a_kwk, [&](auto e){return e < 1;});
    std::cout << "None of            : " << pred3 << " | " << pred3_2 << "\n";


    kwk::copy("simd", b_kwk, a_kwk);
    kwk::copy(c_kwk, a_kwk);
    std::cout << "Copy               :\n" << b_kwk << c_kwk << "\n";

    std::vector<f32> copy (10);
    std::vector<f32> copy2(10);
    
    auto scalar = kwk::view{kwk::source = copy, kwk::of_size(2,5)};
    auto vecto  = kwk::view{kwk::source = copy2, kwk::of_size(2,5)};
    kwk::copy_if("simd", [](auto a){return a < 5;}, vecto, a_kwk);
    kwk::copy_if([](auto a){return a < 5;}, scalar, a_kwk);
    std::cout << "Copy_if <5         :\n" << scalar << vecto << "\n";

    auto find1 = kwk::find("simd", a_kwk, 8);
    auto find1_2 = kwk::find(a_kwk, 8);
    if(find1)
        std::cout << "Find             : "<< find1_2 << " | " << *find1 << "\n";

    auto find1bis = kwk::find_if(a_kwk, [](auto e){return e == 5;});
    auto find2 = kwk::find_if("simd", a_kwk, [](auto e){return e == 5;});
    if(find2)
        std::cout << "Find_if          : "<< find1bis <<  " | " << *find2 << "\n";
    

    auto find3 = kwk::find_if_not("simd", a_kwk, [](auto a){return a <= 4 ;});
    auto find3bis = kwk::find_if_not(a_kwk, [](auto a){return a <= 4;});
    if(find3)
        std::cout << "Find_if_not      : "<< find3bis << " | " << *find3 <<"\n";
    
    auto find4 = kwk::find_last("simd", a_kwk, 7);
    auto find4bis = kwk::find_last(a_kwk, 7);
    if(find4)
        std::cout << "Find_last        : "<< find4bis << " | " << *find4 << "\n";
    
    auto find5 = kwk::find_last_if("simd", a_kwk, [](auto a){return a > 6 && a < 8;});
    auto find5bis = kwk::find_last_if(a_kwk, [](auto a){return a > 6 && a < 8;});
    if(find5)
        std::cout << "Find_last_if     : "<< find5bis << " | " << *find5 << "\n";
    
    auto find6 = kwk::find_last_if_not("simd", a_kwk, [](auto a){return a > 7;});
    auto find6bis = kwk::find_last_if_not(a_kwk, [](auto a){return a > 7;});
    if(find6)
        std::cout << "Find_last_if_not : "<< find6bis << " | " << *find6 << "\n";
    
    auto find7 = kwk::find_first_of("simd", a_kwk, a_kwk);
    auto find7_2 = kwk::find_first_of(a_kwk, a_kwk);
    std::cout << "Find_first_of    : "<< find7_2 << " | " << find7 << "\n";

    kwk::fill("simd", b_kwk, 1.0f);
    kwk::fill(c_kwk, 1.0f);
    std::cout << "Fill             : \n" << c_kwk << b_kwk << "\n";

    kwk::generate("simd", [](auto e){return (10+e) * 5;}, b_kwk);
    kwk::generate([](auto e, auto d){return (10+e+d) * 5;}, c_kwk);
    std::cout << "Generate         :\n" << c_kwk << b_kwk << "\n";

    kwk::iota("simd", b_kwk, 5);
    kwk::iota(c_kwk, 5);
    std::cout << "Iota             :\n" << c_kwk << b_kwk << "\n";

    kwk::iota("simd", b_kwk, 5, 2);
    kwk::iota(c_kwk, 5, 2);
    std::cout << "Iota             :\n" << c_kwk << b_kwk << "\n";
    return 0;
}
