//==================================================================================================
/*
  KIWAKU - Containers Well Made
  Copyright : KIWAKU Contributors & Maintainers
  SPDX-License-Identifier: BSL-1.0
*/
//==================================================================================================
#include "../vecto_extension.hpp"
#include "test.hpp"

TTS_CASE("Check for kwk::reduce(in) 1D")
{
  int data[2];
  int vdata = 1;

  fill_data(data, kwk::of_size(2), true);

  auto d = kwk::view{kwk::source = data, kwk::of_size(2)};

  auto res = kwk::reduce("simd", d);

  TTS_EQUAL(res, vdata);
};

TTS_CASE("Check for kwk::reduce(in) 2D")
{
  int data[2*3];
  int vdata = 36;

  fill_data(data, kwk::of_size(2,3), true);

  auto d = kwk::view{kwk::source = data, kwk::of_size(2,3)};

  auto res = kwk::reduce("simd", d);

  TTS_EQUAL(res, vdata);
};

TTS_CASE("Check for kwk::reduce(in) 3D")
{
  int data[2*3*4];
  int vdata = 1476;

  fill_data(data, kwk::of_size(2,3,4), true);

  auto d = kwk::view{kwk::source = data, kwk::of_size(2,3,4)};

  auto res = kwk::reduce("simd", d);

  TTS_EQUAL(res, vdata);
};

TTS_CASE("Check for kwk::reduce(in) 4D")
{
  int data[2*3*4*5];
  int vdata = 74040;

  fill_data(data, kwk::of_size(2,3,4,5), true);

  auto d = kwk::view{kwk::source = data, kwk::of_size(2,3,4,5)};

  auto res = kwk::reduce("simd", d);

  TTS_EQUAL(res, vdata);
}; 

/* TTS_CASE("Check for kwk::reduce(in, func) 1D")
{
  int data[2];
  int vdata = 10;

  fill_data(data, kwk::of_size(2), true);

  auto d = kwk::view{kwk::source = data, kwk::of_size(2)};

  //int count = 0;
  auto res = reduce("simd", d,std::pair{eve::add, 0});
  [&count](auto a, auto e)
  { 
    //count++;
    return (a+10*e);
  });

  TTS_EQUAL(res,   vdata);
  //TTS_EQUAL(count,   d.numel());
};

TTS_CASE("Check for kwk::reduce(in, func) 2D")
{
  int data[2*3];
  int vdata = 360;

  fill_data(data, kwk::of_size(2,3), true);

  auto d = kwk::view{kwk::source = data, kwk::of_size(2,3)};

  //int count = 0;
  auto res = reduce("simd",d,
  [&count](auto a, auto e)
  { 
    //count++;
    return (a+10*e);
  });

  TTS_EQUAL(res,   vdata);
  //TTS_EQUAL(count,   d.numel());
};

TTS_CASE("Check for kwk::reduce(in, func) 3D")
{
  int data[2*3*4];
  int vdata = 14760;

  fill_data(data, kwk::of_size(2,3,4), true);

  auto d = kwk::view{kwk::source = data, kwk::of_size(2,3,4)};

  //int count = 0;
  auto res = reduce("simd",d,
  [&count](auto a, auto e)
  { 
    //count++;
    return (a+10*e);
  });

  TTS_EQUAL(res,   vdata);
  //TTS_EQUAL(count,   d.numel());
};

TTS_CASE("Check for kwk::reduce(in, func) 4D")
{
  int data[2*3*4*5];
  int vdata = 740400;

  fill_data(data, kwk::of_size(2,3,4,5), true);

  auto d = kwk::view{kwk::source = data, kwk::of_size(2,3,4,5)};

  //int count = 0;
  auto res = reduce("simd", d,
  [&count](auto a, auto e)
  { 
    //count++;
    return (a+10*e);
  });

  TTS_EQUAL(res,   vdata);
  //TTS_EQUAL(count,   d.numel());
}; */

/* TTS_CASE("Check for float kwk::reduce(in, func)")
{
  float data[2*2]      = { 1.f,2.2f
                          , 3.3f,4.4f
                          };

  float vdata        =  10.9f;

  auto d = kwk::view{kwk::source = data, kwk::of_size(2,2)};

  
  //int count = 0;
  auto res = reduce("simd", d, std::pair{eve::add, 0});

  TTS_EQUAL(res,   vdata);

  //TTS_EQUAL(count,   d.numel());
}; */

TTS_CASE("Check for kwk::reduce(in, func, init) 1D")
{
  int data[2];
  int vdata = 11;

  fill_data(data, kwk::of_size(2), true);

  auto d = kwk::view{kwk::source = data, kwk::of_size(2)};

  auto res = reduce("simd", d,std::pair{eve::add, 0}, 10);

  TTS_EQUAL(res,   vdata);
};

TTS_CASE("Check for kwk::reduce(in, func, init) 2D")
{
  int data[2*3];
  int vdata = 136;

  fill_data(data, kwk::of_size(2,3), true);

  auto d = kwk::view{kwk::source = data, kwk::of_size(2,3)};

  auto res = reduce("simd",d,std::pair{eve::add, 0},100);

  TTS_EQUAL(res,   vdata);
};

TTS_CASE("Check for kwk::reduce(in, func, init) 3D")
{
  int data[2*3*4];
  int vdata = 2476;

  fill_data(data, kwk::of_size(2,3,4), true);

  auto d = kwk::view{kwk::source = data, kwk::of_size(2,3,4)};

  auto res = reduce("simd", d,std::pair{eve::add, 0},1000);

  TTS_EQUAL(res,   vdata);
};

TTS_CASE("Check for kwk::reduce(in, func, init) 4D")
{
  int data[2*3*4*5];
  int vdata = 84040;

  fill_data(data, kwk::of_size(2,3,4,5), true);

  auto d = kwk::view{kwk::source = data, kwk::of_size(2,3,4,5)};

  auto res = reduce("simd",d,std::pair{eve::add, 0},10000);

  TTS_EQUAL(res,   vdata);
};