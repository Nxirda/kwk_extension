#include "vecto_extension.hpp"
#include <eve/module/math.hpp>

#include <chrono>
#include <iostream>
#include <vector>
#include <numeric>

using f32 = float;
#define SIZE 10000000

//
template<typename Container>
void bench_transform(Container &a_kwk, Container &b_kwk, Container &c_kwk)
{
    auto start = std::chrono::high_resolution_clock::now();
    kwk::transform("simd", [](auto e, auto e2){return eve::cos(eve::exp(eve::sqrt(e * 1/e2)));}, b_kwk, a_kwk, a_kwk);
    auto stop = std::chrono::high_resolution_clock::now();

    auto start_std = std::chrono::high_resolution_clock::now();
    kwk::transform(        [](auto e, auto e2){return eve::cos(eve::exp(eve::sqrt(e * 1/e2)));}, c_kwk, a_kwk, a_kwk);
    auto stop_std = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
    auto duration_std = std::chrono::duration_cast<std::chrono::microseconds>(stop_std - start_std).count();

    double t_per_elem = static_cast<double>(duration)/SIZE;
    double t_per_elem_std = static_cast<double>(duration_std)/SIZE;

    std::cout << "=== Transform ===\n";
    std::cout << "Simd duration         : " << duration       << " microseconds" << std::endl;
    std::cout << "duration              : " << duration_std   << " microseconds" << std::endl;
    std::cout << "Time per element simd : " << t_per_elem     << " microseconds" << std::endl;
    std::cout << "Time per element      : " << t_per_elem_std << " microseconds" << std::endl;
    std::cout << std::endl;
}

//
template<typename Container>
void bench_reduce(Container &a_kwk, Container &b_kwk)
{
    auto start = std::chrono::high_resolution_clock::now();
    auto res  = kwk::reduce("simd", a_kwk);
    auto stop = std::chrono::high_resolution_clock::now();

    auto start_std = std::chrono::high_resolution_clock::now();
    auto res2 = kwk::reduce(        b_kwk);
    auto stop_std = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
    auto duration_std = std::chrono::duration_cast<std::chrono::microseconds>(stop_std - start_std).count();

    double t_per_elem     = static_cast<double>(duration)/SIZE;
    double t_per_elem_std = static_cast<double>(duration_std)/SIZE;

    std::cout << "=== Reduce ===\n";
    std::cout << "Simd duration         : " << duration       << " microseconds" << std::endl;
    std::cout << "duration              : " << duration_std   << " microseconds" << std::endl;
    std::cout << "Time per element simd : " << t_per_elem     << " microseconds" << std::endl;
    std::cout << "Time per element      : " << t_per_elem_std << " microseconds" << std::endl;
    std::cout << "Res                   : " << res << " | " << res2 << "\n";
    std::cout << std::endl;
}

//
template<typename Container>
void bench_reduce_op(Container &a_kwk, Container &b_kwk)
{
    auto start = std::chrono::high_resolution_clock::now();
    auto res  = kwk::reduce("simd", a_kwk, std::pair{eve::add, 0}         , 2.f);
    auto stop = std::chrono::high_resolution_clock::now();

    auto start_std = std::chrono::high_resolution_clock::now();
    auto res2 = kwk::reduce(        b_kwk, [](auto a, auto b){return a+b;}, 2.f);
    auto stop_std = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
    auto duration_std = std::chrono::duration_cast<std::chrono::microseconds>(stop_std - start_std).count();

    double t_per_elem     = static_cast<double>(duration)/SIZE;
    double t_per_elem_std = static_cast<double>(duration_std)/SIZE;

    std::cout << "=== Reduce Op ===\n";
    std::cout << "Simd duration         : " << duration       << " microseconds" << std::endl;
    std::cout << "duration              : " << duration_std   << " microseconds" << std::endl;
    std::cout << "Time per element simd : " << t_per_elem     << " microseconds" << std::endl;
    std::cout << "Time per element      : " << t_per_elem_std << " microseconds" << std::endl;
    std::cout << "Res                   : " << res << " | " << res2 << "\n";

    std::cout << std::endl;
}

//
template<typename Container>
void bench_transform_reduce(Container &a_kwk, Container &b_kwk, Container &c_kwk)
{
    auto start = std::chrono::high_resolution_clock::now();
    auto res  = kwk::transform_reduce("simd", a_kwk, c_kwk, 0.0, std::pair{eve::add, 0},          [](auto a, auto b){return eve::sqrt(eve::exp(a+b));} );
    auto stop = std::chrono::high_resolution_clock::now();
    
    auto start_std = std::chrono::high_resolution_clock::now();
    auto res2 = kwk::transform_reduce(        a_kwk, b_kwk, 0.0, [](auto a, auto b){return a+b;}, [](auto a, auto b){return eve::sqrt(eve::exp(a+b));} );
    auto stop_std = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
    auto duration_std = std::chrono::duration_cast<std::chrono::microseconds>(stop_std - start_std).count();

    double t_per_elem     = static_cast<double>(duration)/SIZE;
    double t_per_elem_std = static_cast<double>(duration_std)/SIZE;

    std::cout << "=== Transform Reduce ===\n";
    std::cout << "Simd duration         : " << duration       << " microseconds" << std::endl;
    std::cout << "duration              : " << duration_std   << " microseconds" << std::endl;
    std::cout << "Time per element simd : " << t_per_elem     << " microseconds" << std::endl;
    std::cout << "Time per element      : " << t_per_elem_std << " microseconds" << std::endl;
    std::cout << "Res                   : " << res << " | " << res2 << "\n";

    std::cout << std::endl;
}

//
template<typename Container>
void bench_transform_inclusive_scan(Container &a_kwk, Container &b_kwk, Container &c_kwk)
{
    auto start = std::chrono::high_resolution_clock::now();
    kwk::transform_inclusive_scan("simd", a_kwk, c_kwk, 1.f, std::pair{eve::add, 0},          [](auto e){return eve::sqrt(eve::exp(e));});
    auto stop = std::chrono::high_resolution_clock::now();

    auto start_std = std::chrono::high_resolution_clock::now();
    kwk::transform_inclusive_scan(        a_kwk, b_kwk, 1.f, [](auto a, auto b){return a+b;}, [](auto e){return eve::sqrt(eve::exp(e));});
    auto stop_std = std::chrono::high_resolution_clock::now();

    auto duration     = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
    auto duration_std = std::chrono::duration_cast<std::chrono::microseconds>(stop_std - start_std).count();

    double t_per_elem     = static_cast<double>(duration)/SIZE;
    double t_per_elem_std = static_cast<double>(duration_std)/SIZE;

    std::cout << "=== Transform Inclusive Scan ===\n";
    std::cout << "Simd duration         : " << duration       << " microseconds" << std::endl;
    std::cout << "duration              : " << duration_std   << " microseconds" << std::endl;
    std::cout << "Time per element simd : " << t_per_elem     << " microseconds" << std::endl;
    std::cout << "Time per element      : " << t_per_elem_std << " microseconds" << std::endl;

    std::cout << std::endl;
}

//
template<typename Container>
void bench_transform_exclusive_scan(Container &a_kwk, Container &b_kwk, Container &c_kwk)
{
    auto start = std::chrono::high_resolution_clock::now();
    kwk::transform_exclusive_scan("simd", a_kwk, c_kwk, 1.f, std::pair{eve::add, 0},          [](auto e){return eve::sqrt(eve::exp(e));});
    auto stop = std::chrono::high_resolution_clock::now();

    auto start_std = std::chrono::high_resolution_clock::now();
    kwk::transform_exclusive_scan(        a_kwk, b_kwk, 1.f, [](auto a, auto b){return a+b;}, [](auto e){return eve::sqrt(eve::exp(e));});
    auto stop_std = std::chrono::high_resolution_clock::now();

    auto duration     = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
    auto duration_std = std::chrono::duration_cast<std::chrono::microseconds>(stop_std - start_std).count();

    double t_per_elem     = static_cast<double>(duration)/SIZE;
    double t_per_elem_std = static_cast<double>(duration_std)/SIZE;

    std::cout << "=== Transform Exclusive Scan ===\n";
    std::cout << "Simd duration         : " << duration       << " microseconds" << std::endl;
    std::cout << "duration              : " << duration_std   << " microseconds" << std::endl;
    std::cout << "Time per element simd : " << t_per_elem     << " microseconds" << std::endl;
    std::cout << "Time per element      : " << t_per_elem_std << " microseconds" << std::endl;

    std::cout << std::endl;
}

//
template<typename Container>
void bench_all_of(Container &a_kwk)
{
    auto start = std::chrono::high_resolution_clock::now();
    auto pred1   = kwk::all_of("simd", a_kwk, [&](auto e){return e > 2;});
    auto stop = std::chrono::high_resolution_clock::now();

    auto start_std = std::chrono::high_resolution_clock::now();
    auto pred1_2 = kwk::all_of(        a_kwk, [&](auto e){return e > 2;});
    auto stop_std = std::chrono::high_resolution_clock::now();

    auto duration     = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
    auto duration_std = std::chrono::duration_cast<std::chrono::microseconds>(stop_std - start_std).count();

    double t_per_elem     = static_cast<double>(duration)/SIZE;
    double t_per_elem_std = static_cast<double>(duration_std)/SIZE;

    std::cout << "=== All Of ===\n";
    std::cout << "Simd duration         : " << duration       << " microseconds" << std::endl;
    std::cout << "duration              : " << duration_std   << " microseconds" << std::endl;
    std::cout << "Time per element simd : " << t_per_elem     << " microseconds" << std::endl;
    std::cout << "Time per element      : " << t_per_elem_std << " microseconds" << std::endl;
    std::cout << "Res                   : " << pred1 << " | " << pred1_2 << "\n";

    std::cout << std::endl;
}

//
template<typename Container>
void bench_any_of(Container &a_kwk)
{
    auto start = std::chrono::high_resolution_clock::now();
    auto pred2   = kwk::any_of("simd", a_kwk, [&](auto e){return e > SIZE/2;});
    auto stop = std::chrono::high_resolution_clock::now();

    auto start_std = std::chrono::high_resolution_clock::now();
    auto pred2_2 = kwk::any_of(        a_kwk, [&](auto e){return e > SIZE/2;});
    auto stop_std = std::chrono::high_resolution_clock::now();

    auto duration     = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
    auto duration_std = std::chrono::duration_cast<std::chrono::microseconds>(stop_std - start_std).count();

    double t_per_elem     = static_cast<double>(duration)/SIZE;
    double t_per_elem_std = static_cast<double>(duration_std)/SIZE;

    std::cout << "=== Any Of ===\n";
    std::cout << "Simd duration         : " << duration       << " microseconds" << std::endl;
    std::cout << "duration              : " << duration_std   << " microseconds" << std::endl;
    std::cout << "Time per element simd : " << t_per_elem     << " microseconds" << std::endl;
    std::cout << "Time per element      : " << t_per_elem_std << " microseconds" << std::endl;
    std::cout << "Res                   : " << pred2 << " | " << pred2_2 << "\n";
    std::cout << std::endl;

}

//
template<typename Container>
void bench_none_of(Container &a_kwk)
{
    auto start = std::chrono::high_resolution_clock::now();
    auto pred3   = kwk::none_of("simd", a_kwk, [&](auto e){return e > SIZE/2;});
    auto stop = std::chrono::high_resolution_clock::now();

    auto start_std = std::chrono::high_resolution_clock::now();
    auto pred3_2 = kwk::none_of(        a_kwk, [&](auto e){return e > SIZE/2;});
    auto stop_std = std::chrono::high_resolution_clock::now();

    auto duration     = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
    auto duration_std = std::chrono::duration_cast<std::chrono::microseconds>(stop_std - start_std).count();

    double t_per_elem     = static_cast<double>(duration)/SIZE;
    double t_per_elem_std = static_cast<double>(duration_std)/SIZE;

    std::cout << "=== None Of ===\n";
    std::cout << "Simd duration         : " << duration       << " microseconds" << std::endl;
    std::cout << "duration              : " << duration_std   << " microseconds" << std::endl;
    std::cout << "Time per element simd : " << t_per_elem     << " microseconds" << std::endl;
    std::cout << "Time per element      : " << t_per_elem_std << " microseconds" << std::endl;
    std::cout << "Res                   : " << pred3 << " | " << pred3_2 << "\n";
    std::cout << std::endl;
}

//
template<typename Container>
void bench_copy(Container &a_kwk, Container &b_kwk, Container &c_kwk)
{
    auto start = std::chrono::high_resolution_clock::now();
    kwk::copy("simd", b_kwk, a_kwk);
    auto stop = std::chrono::high_resolution_clock::now();

    auto start_std = std::chrono::high_resolution_clock::now();
    kwk::copy(        c_kwk, a_kwk);
    auto stop_std = std::chrono::high_resolution_clock::now();

    auto duration     = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
    auto duration_std = std::chrono::duration_cast<std::chrono::microseconds>(stop_std - start_std).count();

    double t_per_elem     = static_cast<double>(duration)/SIZE;
    double t_per_elem_std = static_cast<double>(duration_std)/SIZE;

    std::cout << "=== Copy ===\n";
    std::cout << "Simd duration         : " << duration       << " microseconds" << std::endl;
    std::cout << "duration              : " << duration_std   << " microseconds" << std::endl;
    std::cout << "Time per element simd : " << t_per_elem     << " microseconds" << std::endl;
    std::cout << "Time per element      : " << t_per_elem_std << " microseconds" << std::endl;
    std::cout << std::endl;
}

//
template<typename Container>
void bench_copy_if(Container &a_kwk, Container &b_kwk, Container &c_kwk)
{
    auto start = std::chrono::high_resolution_clock::now();
    kwk::copy_if("simd", [&](auto e){return e >= 5 && e < SIZE - 15;}, b_kwk, a_kwk);
    auto stop = std::chrono::high_resolution_clock::now();

    auto start_std = std::chrono::high_resolution_clock::now();
    kwk::copy_if(        [&](auto e){return e >= 5 && e < SIZE - 15;}, c_kwk, a_kwk);
    auto stop_std = std::chrono::high_resolution_clock::now();

    auto duration     = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
    auto duration_std = std::chrono::duration_cast<std::chrono::microseconds>(stop_std - start_std).count();

    double t_per_elem     = static_cast<double>(duration)/SIZE;
    double t_per_elem_std = static_cast<double>(duration_std)/SIZE;

    std::cout << "=== Copy_if ===\n";
    std::cout << "Simd duration         : " << duration       << " microseconds" << std::endl;
    std::cout << "duration              : " << duration_std   << " microseconds" << std::endl;
    std::cout << "Time per element simd : " << t_per_elem     << " microseconds" << std::endl;
    std::cout << "Time per element      : " << t_per_elem_std << " microseconds" << std::endl;
    std::cout << std::endl;
}

//
template<typename Container>
void bench_find(Container &a_kwk)
{
    auto start = std::chrono::high_resolution_clock::now();
    auto find1   = kwk::find("simd", a_kwk, SIZE - 10);
    auto stop = std::chrono::high_resolution_clock::now();

    auto start_std = std::chrono::high_resolution_clock::now();
    auto find1_2 = kwk::find(        a_kwk, SIZE - 10);
    auto stop_std = std::chrono::high_resolution_clock::now();

    auto duration     = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
    auto duration_std = std::chrono::duration_cast<std::chrono::microseconds>(stop_std - start_std).count();

    double t_per_elem     = static_cast<double>(duration)/SIZE;
    double t_per_elem_std = static_cast<double>(duration_std)/SIZE;

    std::cout << "=== Find ===\n";
    std::cout << "Simd duration         : " << duration       << " microseconds" << std::endl;
    std::cout << "duration              : " << duration_std   << " microseconds" << std::endl;
    std::cout << "Time per element simd : " << t_per_elem     << " microseconds" << std::endl;
    std::cout << "Time per element      : " << t_per_elem_std << " microseconds" << std::endl;
    if(find1)
        std::cout << "Find             : "<< find1_2 << " | " << *find1 << "\n"; 
    else    
        std::cout << "Find              : " << "Value not found\n";
    std::cout << std::endl;
}

//
template<typename Container>
void bench_find_if(Container &a_kwk)
{
    auto start = std::chrono::high_resolution_clock::now();
    auto find2    = kwk::find_if("simd", a_kwk, [](auto e){return e == SIZE - 10;});
    auto stop = std::chrono::high_resolution_clock::now();

    auto start_std = std::chrono::high_resolution_clock::now();
    auto find     = kwk::find_if(        a_kwk, [](auto e){return e == SIZE - 10;});
    auto stop_std = std::chrono::high_resolution_clock::now();

    auto duration     = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
    auto duration_std = std::chrono::duration_cast<std::chrono::microseconds>(stop_std - start_std).count();

    double t_per_elem     = static_cast<double>(duration)/SIZE;
    double t_per_elem_std = static_cast<double>(duration_std)/SIZE;

    std::cout << "=== Find if ===\n";
    std::cout << "Simd duration         : " << duration       << " microseconds" << std::endl;
    std::cout << "duration              : " << duration_std   << " microseconds" << std::endl;
    std::cout << "Time per element simd : " << t_per_elem     << " microseconds" << std::endl;
    std::cout << "Time per element      : " << t_per_elem_std << " microseconds" << std::endl;
    if(find2)
        std::cout << "Find             : "<< find << " | " << *find2 << "\n"; 
    else    
        std::cout << "Find              : " << "Value not found\n";
    std::cout << std::endl;
}

//
template<typename Container>
void bench_find_if_not(Container &a_kwk)
{
    auto start = std::chrono::high_resolution_clock::now();
    auto find3 = kwk::find_if_not("simd", a_kwk, [](auto a){return a < SIZE - 10 ;});
    auto stop = std::chrono::high_resolution_clock::now();

    auto start_std = std::chrono::high_resolution_clock::now();
    auto find  = kwk::find_if_not(        a_kwk, [](auto a){return a < SIZE - 10;});
    auto stop_std = std::chrono::high_resolution_clock::now();

    auto duration     = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
    auto duration_std = std::chrono::duration_cast<std::chrono::microseconds>(stop_std - start_std).count();

    double t_per_elem     = static_cast<double>(duration)/SIZE;
    double t_per_elem_std = static_cast<double>(duration_std)/SIZE;

    std::cout << "=== Find if not ===\n";
    std::cout << "Simd duration         : " << duration       << " microseconds" << std::endl;
    std::cout << "duration              : " << duration_std   << " microseconds" << std::endl;
    std::cout << "Time per element simd : " << t_per_elem     << " microseconds" << std::endl;
    std::cout << "Time per element      : " << t_per_elem_std << " microseconds" << std::endl;
    if(find3)
        std::cout << "Find             : "<< find << " | " << *find3 << "\n"; 
    else    
        std::cout << "Find              : " << "Value not found\n";
    std::cout << std::endl;
}

//
template<typename Container>
void bench_find_last(Container &a_kwk)
{
    auto start = std::chrono::high_resolution_clock::now();
    auto find4 = kwk::find_last("simd", a_kwk, 10);
    auto stop = std::chrono::high_resolution_clock::now();

    auto start_std = std::chrono::high_resolution_clock::now();
    auto find  = kwk::find_last(        a_kwk, 10);
    auto stop_std = std::chrono::high_resolution_clock::now();

    auto duration     = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
    auto duration_std = std::chrono::duration_cast<std::chrono::microseconds>(stop_std - start_std).count();

    double t_per_elem     = static_cast<double>(duration)/SIZE;
    double t_per_elem_std = static_cast<double>(duration_std)/SIZE;

    std::cout << "=== Find last ===\n";
    std::cout << "Simd duration         : " << duration       << " microseconds" << std::endl;
    std::cout << "duration              : " << duration_std   << " microseconds" << std::endl;
    std::cout << "Time per element simd : " << t_per_elem     << " microseconds" << std::endl;
    std::cout << "Time per element      : " << t_per_elem_std << " microseconds" << std::endl;
    if(find4)
        std::cout << "Find             : "<< find << " | " << *find4 << "\n"; 
    else    
        std::cout << "Find              : " << "Value not found\n";
    std::cout << std::endl;
}

//
template<typename Container>
void bench_find_last_if(Container &a_kwk)
{
    auto start = std::chrono::high_resolution_clock::now();
    auto find5 = kwk::find_last_if("simd", a_kwk, [](auto a){return a == 10;});
    auto stop = std::chrono::high_resolution_clock::now();

    auto start_std = std::chrono::high_resolution_clock::now();
    auto find  = kwk::find_last_if(        a_kwk, [](auto a){return a == 10;});
    auto stop_std = std::chrono::high_resolution_clock::now();

    auto duration     = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
    auto duration_std = std::chrono::duration_cast<std::chrono::microseconds>(stop_std - start_std).count();

    double t_per_elem     = static_cast<double>(duration)/SIZE;
    double t_per_elem_std = static_cast<double>(duration_std)/SIZE;

    std::cout << "=== Find last if ===\n";
    std::cout << "Simd duration         : " << duration       << " microseconds" << std::endl;
    std::cout << "duration              : " << duration_std   << " microseconds" << std::endl;
    std::cout << "Time per element simd : " << t_per_elem     << " microseconds" << std::endl;
    std::cout << "Time per element      : " << t_per_elem_std << " microseconds" << std::endl;
    if(find5)
        std::cout << "Find             : "<< find << " | " << *find5 << "\n"; 
    else    
        std::cout << "Find              : " << "Value not found\n";
    std::cout << std::endl;
}

//
template<typename Container>
void bench_find_last_if_not(Container &a_kwk)
{
    auto start = std::chrono::high_resolution_clock::now();
    auto find6 = kwk::find_last_if_not("simd", a_kwk, [](auto a){return a > 2;});
    auto stop = std::chrono::high_resolution_clock::now();

    auto start_std = std::chrono::high_resolution_clock::now();
    auto find  = kwk::find_last_if_not(        a_kwk, [](auto a){return a > 2;});
    auto stop_std = std::chrono::high_resolution_clock::now();

    auto duration     = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
    auto duration_std = std::chrono::duration_cast<std::chrono::microseconds>(stop_std - start_std).count();

    double t_per_elem     = static_cast<double>(duration)/SIZE;
    double t_per_elem_std = static_cast<double>(duration_std)/SIZE;

    std::cout << "=== Find last if not ===\n";
    std::cout << "Simd duration         : " << duration       << " microseconds" << std::endl;
    std::cout << "duration              : " << duration_std   << " microseconds" << std::endl;
    std::cout << "Time per element simd : " << t_per_elem     << " microseconds" << std::endl;
    std::cout << "Time per element      : " << t_per_elem_std << " microseconds" << std::endl;
    if(find6)
        std::cout << "Find             : "<< find << " | " << *find6 << "\n"; 
    else    
        std::cout << "Find              : " << "Value not found\n";
    std::cout << std::endl;
}

//
template<typename Container>
void bench_find_first_of(Container &a_kwk, Container &b_kwk, Container &c_kwk)
{
    auto start = std::chrono::high_resolution_clock::now();
    auto find7 = kwk::find_first_of("simd", a_kwk, b_kwk);
    auto stop = std::chrono::high_resolution_clock::now(); 

    auto start_std = std::chrono::high_resolution_clock::now();
    auto find  = kwk::find_first_of(        a_kwk, c_kwk);
    auto stop_std = std::chrono::high_resolution_clock::now();

    auto duration     = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
    auto duration_std = std::chrono::duration_cast<std::chrono::microseconds>(stop_std - start_std).count();

    double t_per_elem     = static_cast<double>(duration)/SIZE;
    double t_per_elem_std = static_cast<double>(duration_std)/SIZE;

    std::cout << "=== Find first of ===\n";
    std::cout << "Simd duration         : " << duration       << " microseconds" << std::endl;
    std::cout << "duration              : " << duration_std   << " microseconds" << std::endl;
    std::cout << "Time per element simd : " << t_per_elem     << " microseconds" << std::endl;
    std::cout << "Time per element      : " << t_per_elem_std << " microseconds" << std::endl;
    if(find7)
        std::cout << "Find              : " << find << " | " << *find7 << "\n"; 
    else    
        std::cout << "Find              : " << "Value not found\n";
    std::cout << std::endl; 
}

//
template<typename Container>
void bench_fill(Container &a_kwk)
{
    auto start = std::chrono::high_resolution_clock::now();
    kwk::fill("simd", a_kwk, 1.0f/80.0);
    auto stop = std::chrono::high_resolution_clock::now();

    auto start_std = std::chrono::high_resolution_clock::now();
    kwk::fill(        a_kwk, 1.0f/80.0);
    auto stop_std = std::chrono::high_resolution_clock::now();

    auto duration     = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
    auto duration_std = std::chrono::duration_cast<std::chrono::microseconds>(stop_std - start_std).count();

    double t_per_elem     = static_cast<double>(duration)/SIZE;
    double t_per_elem_std = static_cast<double>(duration_std)/SIZE;

    std::cout << "=== Fill ===\n";
    std::cout << "Simd duration         : " << duration       << " microseconds" << std::endl;
    std::cout << "duration              : " << duration_std   << " microseconds" << std::endl;
    std::cout << "Time per element simd : " << t_per_elem     << " microseconds" << std::endl;
    std::cout << "Time per element      : " << t_per_elem_std << " microseconds" << std::endl;

    std::cout << std::endl;
}

//
template<typename Container>
void bench_iota(Container &a_kwk)
{
    auto start = std::chrono::high_resolution_clock::now();
    kwk::iota("simd", a_kwk, 5);
    auto stop = std::chrono::high_resolution_clock::now();

    auto start_std = std::chrono::high_resolution_clock::now();
    kwk::iota(        a_kwk, 3);
    auto stop_std = std::chrono::high_resolution_clock::now();

    auto duration     = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
    auto duration_std = std::chrono::duration_cast<std::chrono::microseconds>(stop_std - start_std).count();

    double t_per_elem     = static_cast<double>(duration)/SIZE;
    double t_per_elem_std = static_cast<double>(duration_std)/SIZE;

    std::cout << "=== Iota ===\n";
    std::cout << "Simd duration         : " << duration       << " microseconds" << std::endl;
    std::cout << "duration              : " << duration_std   << " microseconds" << std::endl;
    std::cout << "Time per element simd : " << t_per_elem     << " microseconds" << std::endl;
    std::cout << "Time per element      : " << t_per_elem_std << " microseconds" << std::endl;

    std::cout << std::endl;
}

//
template<typename Container>
void bench_iota_step(Container &a_kwk)
{
    auto start = std::chrono::high_resolution_clock::now();
    kwk::iota("simd", a_kwk, 5, 2);
    auto stop = std::chrono::high_resolution_clock::now();

    auto start_std = std::chrono::high_resolution_clock::now();
    kwk::iota(        a_kwk, 3, 2);
    auto stop_std = std::chrono::high_resolution_clock::now();

    auto duration     = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
    auto duration_std = std::chrono::duration_cast<std::chrono::microseconds>(stop_std - start_std).count();

    double t_per_elem     = static_cast<double>(duration)/SIZE;
    double t_per_elem_std = static_cast<double>(duration_std)/SIZE;

    std::cout << "=== Iota step ===\n";
    std::cout << "Simd duration         : " << duration       << " microseconds" << std::endl;
    std::cout << "duration              : " << duration_std   << " microseconds" << std::endl;
    std::cout << "Time per element simd : " << t_per_elem     << " microseconds" << std::endl;
    std::cout << "Time per element      : " << t_per_elem_std << " microseconds" << std::endl;

    std::cout << std::endl;
}

//
int main(int argc, char **argv)
{
    if(argc == 0)
    {
        std::cout <<"Usage is " << argv[0] << "\n";
    }

    std::vector<f32> a(SIZE);
    std::vector<f32> b(SIZE);
    std::vector<f32> c(SIZE);

    auto shp = kwk::of_size(2, SIZE/2);

    std::iota(a.begin(), a.end(), 0.f);

    auto a_kwk = kwk::view{kwk::source = a, shp};
    auto b_kwk = kwk::view{kwk::source = b, shp};
    auto c_kwk = kwk::view{kwk::source = c, shp};

    // Perfs tests
    bench_transform(a_kwk, b_kwk, c_kwk);
    
    std::iota(b.begin(), b.end(), 0.f);
    std::iota(c.begin(), c.end(), 0.f);
    bench_reduce(b_kwk, c_kwk);

    std::iota(b.begin(), b.end(), 0.f);
    std::iota(c.begin(), c.end(), 0.f);
    bench_reduce_op(b_kwk, c_kwk);

    bench_transform_reduce(a_kwk, b_kwk, c_kwk);

    bench_transform_exclusive_scan(a_kwk, b_kwk, c_kwk);

    bench_transform_inclusive_scan(a_kwk, b_kwk, c_kwk);

    bench_all_of(a_kwk);
    
    bench_any_of(a_kwk);

    bench_none_of(a_kwk);

    bench_find(a_kwk);
    bench_find_if(a_kwk);
    bench_find_if_not(a_kwk);
    
    bench_find_last(a_kwk);
    bench_find_last_if(a_kwk);
    bench_find_last_if_not(a_kwk);

    /* std::iota(b.begin(), b.end(), SIZE/2);
    std::iota(c.begin(), c.end(), SIZE/2);
    bench_find_first_of(a_kwk, b_kwk, c_kwk); */

    bench_fill(a_kwk);

    bench_iota(a_kwk);
    
    bench_iota_step(a_kwk);

    bench_copy(a_kwk, b_kwk, c_kwk);
    bench_copy_if(a_kwk, b_kwk, c_kwk);
    /* std::vector<f32> copy (SIZE);
    std::vector<f32> copy2(SIZE);
     */
   

    return 0;
}

 /* 
    auto scalar = kwk::view{kwk::source = copy,  kwk::of_size(2,5)};
    auto vecto  = kwk::view{kwk::source = copy2, kwk::of_size(2,5)};
    kwk::copy_if("simd", [](auto a){return a > 5;}, vecto,  a_kwk);
    kwk::copy_if(        [](auto a){return a > 5;}, scalar, a_kwk);
    std::cout << "Copy_if <5         :\n" << scalar << vecto << "\n"; */

    /* kwk::generate("simd", [](auto e){return (10+e) * 5;}, b_kwk);
    kwk::generate(        [](auto e, auto d){return (10+e+d) * 5;}, c_kwk);
    std::cout << "Generate         :\n" << c_kwk << b_kwk << "\n"; 
*/
