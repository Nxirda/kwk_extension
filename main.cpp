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

    auto a_kwk = kwk::view{kwk::source = a, kwk::of_size(10)};
    auto b_kwk = kwk::view{kwk::source = out_transform, kwk::of_size(10)};
    auto c_kwk = kwk::view{kwk::source = out_copy, kwk::of_size(10)};
    
    std::cout << "a_kwk : " << a_kwk << "\n";
    std::cout << "b_kwk : " << b_kwk << "\n";

    kwk::transform("simd", [](auto e, auto b){return e * 5.f + b ;}, b_kwk, a_kwk, a_kwk);
    std::cout << "Transform     : " << b_kwk;
    kwk::transform([](auto e, auto b){return e * 5.f + b;}, b_kwk, a_kwk, a_kwk);
    std::cout << "Transform kwk : " << b_kwk << "\n";


    auto res = kwk::reduce("simd", b_kwk);
    std::cout << "Reduction         : " << res << "\n";
    auto res2 = kwk::reduce(b_kwk);
    std::cout << "Reduction kwk     : " << res2 << "\n";

    
    auto res_reduce = kwk::reduce("simd", b_kwk, std::pair{eve::add, 0}, 2.f);
    std::cout << "Reduction with op :" << res_reduce << "\n";
    auto res_reduce2 = kwk::reduce(b_kwk, [](auto a, auto b){return a+b;}, 2.f);
    std::cout << "Reduction kwk     :" << res_reduce2 << "\n";

    auto res_trs = kwk::transform_reduce("simd", b_kwk, b_kwk, 0.0, std::pair{eve::add, 0}, [](auto a, auto b){return a+b;} );
    std::cout << "Transfor_reduce   :" << res_trs << "\n";
    auto res_trs2 = kwk::transform_reduce(b_kwk, b_kwk, 0.0, [](auto a, auto b){return a+b;}, [](auto a, auto b){return a+b;} );
    std::cout << "Transfor_reduce   :" << res_trs2 << "\n";


    auto pred1 = kwk::all_of("simd", a_kwk, [&](auto e){return e > 2;});
    std::cout << "All of             : " << pred1 << "\n";
    auto pred1_2 = kwk::all_of(a_kwk, [&](auto e){return e > 2;});
    std::cout << "All of             : " << pred1_2 << "\n";

    auto pred2 = kwk::any_of("simd", a_kwk, [&](auto e){return e < 5;});
    std::cout << "Any of             : " << pred2 << "\n";
    auto pred2_2 = kwk::any_of(a_kwk, [&](auto e){return e < 5;});
    std::cout << "Any of             : " << pred2_2 << "\n";

    auto pred3 = kwk::none_of("simd", a_kwk, [&](auto e){return e < 1;});
    std::cout << "None of            : " << pred3 << "\n";
    auto pred3_2 = kwk::none_of(a_kwk, [&](auto e){return e < 1;});
    std::cout << "None of            : " << pred3_2 << "\n";


    kwk::copy("simd", b_kwk, a_kwk);
    std::cout << "Copy               :\n" << b_kwk << "\n";
    kwk::copy(b_kwk, a_kwk);
    std::cout << "Copy               :\n" << b_kwk << "\n";

    kwk::copy_if("simd", [](auto a){return a < 5;}, c_kwk, a_kwk);
    std::cout << "Copy_if <5         :\n" << c_kwk << "\n";
    kwk::copy_if([](auto a){return a < 5;}, c_kwk, a_kwk);
    std::cout << "Copy_if <5         :\n" << c_kwk << "\n";

    auto find1 = kwk::find("simd", a_kwk, 5);
    std::cout << "Find             : "<< find1 << "\n";
    auto find1_2 = kwk::find(a_kwk, 5);
    std::cout << "Find kwk         : "<< find1_2 << "\n";

    auto find1bis = kwk::find_if(a_kwk, [](auto e){return e == 5;});
    std::cout << "Find_if kwk         : "<< find1bis << "\n";
    auto find2 = kwk::find_if("simd", a_kwk, [](auto a){return a > 4 && a < 6;});
    std::cout << "Find_if          : "<< find2 << "\n";

    auto find3 = kwk::find_if_not("simd", a_kwk, [](auto a){return a <= 4 ;});
    std::cout << "Find_if_not      : "<< find3 <<"\n";
    auto find3bis = kwk::find_if_not(a_kwk, [](auto a){return a <= 4;});
    std::cout << "Find_if_not kwk  : "<< find3bis << "\n";
    std::cout << std::endl;
    
    auto find4 = kwk::find_last("simd", a_kwk, 7);
    std::cout << "Find_last        : "<< find4 << "\n";
    auto find4bis = kwk::find_last(a_kwk, 7);
    std::cout << "Find_last kwk    : "<<find4bis << "\n";
    std::cout << std::endl;
    
    auto find5 = kwk::find_last_if("simd", a_kwk, [](auto a){return a > 6 && a < 8;});
    std::cout << "Find_last_if     : "<< find5 << "\n";
    auto find5bis = kwk::find_last_if(a_kwk, [](auto a){return a > 6 && a < 8;});
    std::cout << "Find_last_if kwk : " << find5bis << "\n";
    std::cout << std::endl;
    
    auto find6 = kwk::find_last_if_not("simd", a_kwk, [](auto a){return a > 7;});
    std::cout << "Find_last_if_not : "<< find6 << "\n";
    auto find6bis = kwk::find_last_if_not(a_kwk, [](auto a){return a > 7;});
    std::cout << "Find_last_if_not : "<<find6bis << "\n";
    std::cout << std::endl;
    
    auto find7 = kwk::find_first_of("simd", a_kwk, b_kwk);
    std::cout << "Find_first_of    : "<< find7 <<"\n";
    auto find7_2 = kwk::find_first_of(a_kwk, b_kwk);
    std::cout << "Find_first_of    : "<< find7_2 <<"\n";

    return 0;
}
