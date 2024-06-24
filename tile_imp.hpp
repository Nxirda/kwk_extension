#include <kwk/kwk.hpp>

using usz = std::size_t;

//
template<typename Base>
struct tiler : Base
{
  using parent = Base;
  using type_t = typename Base::value_type;
  using test_t = typename Base::shape_type;

  KWK_TRIVIAL constexpr tiler(Base b) : parent(b)
  {}
 
  constexpr auto pattern_shape()    const 
  { 
    constexpr usz point = parent::static_order/2;
    auto shp = kumi::get<1>(kumi::split(parent::shape(),kumi::index<point>)); 
    return kwk::of_size(shp);
  }

  constexpr auto shape()    const 
  { 
    constexpr usz point = parent::static_order/2;
    auto shp = kumi::get<0>(kumi::split(parent::shape(),kumi::index<point>)); 
    return kwk::of_size(shp);
  }
  
  constexpr auto stride()   const 
  { 
    constexpr usz point = parent::static_order/2;
    auto strd = kumi::get<0>(kumi::split(parent::stride(),kumi::index<point>)); 
    return kwk::with_strides(strd);
  }
 
  constexpr auto get_data() const { return parent::get_data(); }
  constexpr auto get_data()       { return parent::get_data(); }

  auto operator()(auto i1, auto i0) const noexcept 
  {
    using namespace kwk::literals;
    const auto [t1, t0] = this->pattern_shape();
    
    auto pos = kumi::make_tuple(i1,i0,0,0);
    
    return  kwk::view{  kwk::source = &parent::operator()(pos)
                      , kwk::of_size(1_c,1_c, t1,t0 ) 
                      , kwk::strides = parent::stride() };
  }
};

template <kwk::concepts::container Container>
auto overlapping_tiles(Container const& c, auto const& shp)
{
  auto tsz = kumi::apply([](auto... m)
            {
                using kumi::get;
                return kwk::of_size(( get<0>(m) - get<1>(m)+1)...
                                    , get<1>(m)...);

            },kumi::zip(c.shape(), shp));

  auto base = kwk::view{ kwk::source = c.get_data()
                       , tsz
                       , kwk::strides = kwk::with_strides(kumi::cat(c.stride(), c.stride()))
                       };
  return tiler{base};
}

template <kwk::concepts::container Container>
auto paving_tiles(Container const& c, auto const& shp, auto const& offset)
{
  auto tsz = kumi::apply([](auto... m)
            {
              using kumi::get;
              return kwk::of_size((get<0>(m) - get<1>(m)) / get<2>(m) +1 ...
                                  ,get<1>(m)...); 

            }, kumi::zip(c.shape(), shp, offset));

  auto tst = kumi::apply([](auto... m)
            {
              using kumi::get;
              return kwk::with_strides( get<0>(m) * get<1>(m) ...,
                                        get<0 >(m)...); 

            }, kumi::zip(c.stride(), offset));

  auto base = kwk::view{  kwk::source = c.get_data(),
                          tsz, 
                          kwk::strides = tst
                       };
  return tiler{base};
}

//
/* template<kwk::concepts::container Container>
auto overlapping_tiles(Container const &c, auto const &shp, auto const &stride)
{

    auto out_shape  = kumi::reverse(c.shape());
    auto low_stride = kumi::reverse(shp);
    auto out_stride = kumi::reverse(stride);

    auto product = 1;
    kumi::for_each([&](auto &res, auto &low_str, auto &acc_str){
        auto save = res;
        res = (res-low_str)/acc_str +1; 
        low_str = product;  
        product *= save;
        acc_str *= low_str;

    }, out_shape, low_stride, out_stride);
    
    out_shape = kumi::reverse(out_shape);
    
    auto tile_shape  = kumi::cat(out_shape, shp);
    auto tile_stride = kumi::cat(out_stride, low_stride);
    tile_stride = kumi::reverse(tile_stride);

    auto base = kwk::view{  kwk::source = c.get_data(), 
                            kwk::of_size(tile_shape),
                            kwk::with_strides(tile_stride) };
    return tiler{base};
} */

//
/* auto tiles(auto const& c, auto const& shp)
{
    return overlapping_tiles(c, shp, kwk::with_strides(shp));
} */