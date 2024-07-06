#pragma once

#include <kumi/tuple.hpp>
#include <kwk/kwk.hpp>
#include <eve/eve.hpp>
#include <eve/module/algo.hpp> 

namespace kwk
{

    //Simple function that builds a range on a kwk container
    template <typename Ptr>
    auto make_range(Ptr p, auto n)
    {
        return eve::algo::as_range(p, p+n);
    }


    /********************* Transform  *********************************/
    // Context signature isnt correct ofc (simplified)
    template<typename Context, typename Func, concepts::container Out
            ,concepts::container C0, concepts::container... Cs
            >
    constexpr auto transform([[maybe_unused]] Context &ctx, Func &&f, Out& out, C0 const& c0, Cs const&... cs)
    {
        if constexpr (Out::preserve_reachability && C0::preserve_reachability && (Cs::preserve_reachability && ...))
        {   
            auto sz = out.numel();
            auto r_out = make_range(out.get_data(), sz);
            auto zipped = eve::views::zip(make_range(c0.get_data(), sz), make_range(cs.get_data(), sz)...);
            
            eve::algo::transform_to(zipped, r_out, [&f](auto const &in){return kumi::apply(f, in);});
        }
        else 
        {
            auto s   = kumi::split(c0.shape(), kumi::index<C0::static_order -1>);
            auto ext = kumi::get<0>(s);
            auto in  = kumi::get<0>(kumi::get<1>(s));
            kwk::__::for_each([&](auto... is)
            {
                auto r_out = make_range(&out(is...,0), in);
                auto zipped = eve::views::zip(make_range(&c0(is...,0), in), make_range(&cs(is...,0), in)...);
                eve::algo::transform_to(zipped, r_out, [&f](auto const &in){return kumi::apply(f, in);});
            }, ext);
        }
    }
   
    /********************* Reductions *********************************/
    //
    template<typename Context, concepts::container In>
    constexpr auto reduce([[maybe_unused]] Context &ctx, In const& in)
    {
        if constexpr (In::preserve_reachability)
        {
            auto r_in = make_range(in.get_data(), in.numel());
            return eve::algo::reduce(r_in, typename In::value_type{});
        }
        else
        {
            auto s   = kumi::split(in.shape(), kumi::index<In::static_order -1>);
            auto ext = kumi::get<0>(s);                             
            auto inn = kumi::get<0>(kumi::get<1>(s));               
            auto acc = 0;                                           
            kwk::__::for_each([&](auto... is)
            {
                auto r_in = make_range(&in(is...,0), inn);
                acc = eve::algo::reduce(r_in, acc);
            }, ext);
            
            return acc;
        }
    }

    //     
    template<typename Context, typename Op, typename Id, concepts::container In>
    constexpr auto reduce([[maybe_unused]] Context& ctx, In const& in, std::pair<Op, Id> R, auto init)
    {
        if constexpr (In::preserve_reachability)
        {
            auto r_in = make_range(in.get_data(), in.numel());
            return eve::algo::reduce(r_in, R, init);
        }
        else
        {
            auto s   = kumi::split(in.shape(), kumi::index<In::static_order -1>);
            auto ext = kumi::get<0>(s);                             
            auto inn = kumi::get<0>(kumi::get<1>(s));               
            auto acc = init;                                           
            kwk::__::for_each([&](auto... is)
            {
                auto r_in = make_range(&in(is...,0), inn);
                acc = eve::algo::reduce(r_in, R, acc);
            }, ext);
            
            return acc;
        }
    }

    /********************* Numeric *********************************/

    // 
    template<typename Context, typename Op, typename Id, typename Func_T, 
            concepts::container In>
    constexpr auto transform_reduce([[maybe_unused]] Context& ctx, In const& in1, In const& in2, auto init, std::pair<Op, Id> R, Func_T T)
    {
        if constexpr (In::preserve_reachability)
        {
            auto r_in1 = make_range(in1.get_data(), in1.numel());
            auto r_in2 = make_range(in2.get_data(), in2.numel());
            auto zipped = eve::views::zip(r_in1, r_in2);

            return eve::algo::transform_reduce(zipped, [&T](auto const& in){return kumi::apply(T, in);}, R, init);
        }
        else
        {
            auto s   = kumi::split(in1.shape(), kumi::index<In::static_order -1>);
            auto ext = kumi::get<0>(s);                             
            auto inn = kumi::get<0>(kumi::get<1>(s));               
            auto acc = init;                                           
            kwk::__::for_each([&](auto... is)
            {
                auto r_in1 = make_range(&in1(is...,0), inn);
                auto r_in2 = make_range(&in2(is...,0), inn);
                auto zipped = eve::views::zip(r_in1, r_in2);

                acc = eve::algo::transform_reduce(zipped, [&T](auto const& in){return kumi::apply(T, in);}, R, acc);
   
            }, ext);
            
            return acc;
        }
    }
  
    //
    template<typename Context, typename Op, typename Id, typename Func_T, 
            concepts::container In>
    constexpr auto inner_product([[maybe_unused]] Context& ctx, In const& in1, In const& in2, auto init, std::pair<Op, Id> R, Func_T T)
    {
        return kwk::transform_reduce(ctx, in1, in2, init, R, T);
    }

    //
    template<typename Context, typename Op, typename Id, typename Func_T,
            concepts::container In, concepts::container Out>
    constexpr auto transform_inclusive_scan([[maybe_unused]] Context& ctx, In const& in, Out& out, auto init, std::pair<Op, Id> S, Func_T T)
    {
        kwk::transform("simd", T, out, in);
      
        if constexpr (Out::preserve_reachability && In::preserve_reachability)
        {
            auto r_out = make_range(out.get_data(), out.numel());
            eve::algo::inclusive_scan_inplace(r_out, S, init);
        }
        else
        {
            auto s   = kumi::split(in.shape(), kumi::index<In::static_order -1>);
            auto ext = kumi::get<0>(s);               
            auto inn = kumi::get<0>(kumi::get<1>(s)); 
            kwk::__::for_each([&](auto... is)
            {
                auto r_out = make_range(&out(is...,0), inn);
                eve::algo::inclusive_scan_inplace(r_out, S, init);
            }, ext);
        }
    }

    //
    template<typename Context, typename Op, typename Id, typename Func_T,
            concepts::container In, concepts::container Out>
    constexpr auto transform_exclusive_scan([[maybe_unused]] Context& ctx, In const& in, Out &out, auto init, std::pair<Op, Id> S, Func_T T)
    {
        kwk::transform("simd", T, out, in);
        
        if constexpr (Out::preserve_reachability && In::preserve_reachability)
        {
            
            *out.get_data() = init;
            auto r_in  = make_range(in.get_data(), in.numel() -1);
            // rr : rotate right
            auto r_outrr = make_range(out.get_data() +1, out.numel() -1);
            eve::algo::inclusive_scan_to(r_in, r_outrr, S, init);
        }
        else
        {  
            auto s   = kumi::split(in.shape(), kumi::index<In::static_order -1>);
            auto ext = kumi::get<0>(s);               
            auto inn = kumi::get<0>(kumi::get<1>(s)); 
            *out.get_data() = init;
            kwk::__::for_each([&](auto... is)
            {
                auto r_in = make_range(&in(is...,0), inn -1);
                // rr : rotate right
                auto r_outrr = make_range(&in(is...,0) +1, inn -1);
                eve::algo::inclusive_scan_to(r_in, r_outrr, S, init);

            }, ext);
        }        
    }

    /************************ Copy *********************************/
    //
    template<typename Context, concepts::container Out, concepts::container In>
    constexpr auto copy([[maybe_unused]] Context& ctx, Out& out, In const& in)
    {
        if constexpr (Out::preserve_reachability && In::preserve_reachability)
        {
            auto r_in  = make_range(in.get_data(), in.numel());
            auto r_out = make_range(out.get_data(), out.numel());
            eve::algo::copy(r_in, r_out);
        }
        else
        {
            auto s   = kumi::split(in.shape(), kumi::index<In::static_order -1>);
            auto ext = kumi::get<0>(s);                
            auto inn = kumi::get<0>(kumi::get<1>(s)); 
            kwk::__::for_each([&](auto... is)
            {
                auto r_in  = make_range(&in(is...,0), inn);
                auto r_out = make_range(&out(is...,0), inn);
                eve::algo::copy(r_in, r_out);
            }, ext);
        }
    }

    // Shall be eve::transform_copy_if
    template<typename Context, typename Func,
             concepts::container Out, concepts::container In
            >
    constexpr auto copy_if([[maybe_unused]] Context& ctx, Func f, Out& out, In const& in)
    {
        if constexpr (Out::preserve_reachability && In::preserve_reachability)
        {   
            auto r_in  = make_range(in.get_data(), in.numel());
            auto r_out = make_range(out.get_data(), out.numel());
            eve::algo::copy_if(r_in, r_out, f);
        }
        else
        {
            auto s   = kumi::split(in.shape(), kumi::index<In::static_order -1>);
            auto ext = kumi::get<0>(s);                             
            auto inn = kumi::get<0>(kumi::get<1>(s));    
            kwk::__::for_each([&](auto... is)
            {
                auto r_in  = make_range(&in(is...,0), inn);
                auto r_out = make_range(&out(is...,0), inn);
                eve::algo::copy_if(r_in, r_out, f);
            }, ext);
        }
    }

    /********************* Predicates *********************************/
    // 
    template<typename Context, typename Func,
             concepts::container In
            >
    constexpr auto all_of([[maybe_unused]] Context& ctx, In const& in, Func f)
    {
        if constexpr (In::preserve_reachability)
        {
            auto r_in = make_range(in.get_data(), in.numel());
            return eve::algo::all_of(r_in, f);
        }
        else
        {
            auto s   = kumi::split(in.shape(), kumi::index<In::static_order -1>);
            auto ext = kumi::get<0>(s);                
            auto inn = kumi::get<0>(kumi::get<1>(s));  
            bool b   = true;

            kwk::__::for_each([&](auto... is)
            {
                auto r_in = make_range(&in(is...,0), inn);
                b = b & eve::algo::all_of(r_in, f);
            }, ext);

            return b;
        }
    }

    //
    template<typename Context, typename Func,
             concepts::container In
            >
    constexpr auto any_of([[maybe_unused]] Context& ctx, In const& in, Func f)
    {
        if constexpr (In::preserve_reachability)
        {
            auto r_in = make_range(in.get_data(), in.numel());
            return eve::algo::any_of(r_in, f);
        }
        else
        {
            auto s   = kumi::split(in.shape(), kumi::index<In::static_order -1>);
            auto ext = kumi::get<0>(s);                
            auto inn = kumi::get<0>(kumi::get<1>(s)); 
            bool b   = true;

            kwk::__::for_each([&](auto... is)
            {
                auto r_in = make_range(&in(is...,0), inn);
                b = b & eve::algo::any_of(r_in, f);
            }, ext);

            return b;
        }
    }

    //
    template<typename Context, typename Func,
             concepts::container In
            >
    constexpr auto none_of([[maybe_unused]] Context& ctx, In const& in, Func f)
    {
        if constexpr (In::preserve_reachability)
        {
            auto r_in = make_range(in.get_data(), in.numel());
            return eve::algo::none_of(r_in, f);
        }
        else
        {
            auto s   = kumi::split(in.shape(), kumi::index<In::static_order -1>);
            auto ext = kumi::get<0>(s);              
            auto inn = kumi::get<0>(kumi::get<1>(s)); 
            bool b   = true;

            kwk::__::for_each([&](auto... is)
            {
                auto r_in = make_range(&in(is...,0), inn);
                b = b & eve::algo::none_of(r_in, f);
            }, ext);

            return b;
        }
    }

    /*
        Count && count if arent specific simd algos, they just need to call
        the standard algo with the simd context
    */

    /********************* Finds *********************************/ 
    //
    template<typename Context, concepts::container In>
    constexpr auto find([[maybe_unused]] Context& ctx, In const& in, auto value)
    {
        return find_if("simd", in, [&](auto e){return e == value;});
    }

    //
    template<typename Context, concepts::container In, typename Func>
    constexpr auto find_if([[maybe_unused]] Context& ctx, In const& in, Func f)
    {
        auto c = kumi::generate<In::static_order, std::ptrdiff_t>(0);
        using coords_t = decltype(c);

        if constexpr (In::preserve_reachability)
        {
            auto r_in = make_range(in.get_data(), in.numel());
            auto find = eve::algo::find_if(r_in, f);
            if(find < r_in.end())
            {
                auto linear_pos  = find - r_in.begin();
                auto pos = kwk::coordinates(linear_pos, in.shape());
                return std::optional<coords_t>{kumi::to_tuple(pos)};
            }
            return std::optional<coords_t>{std::nullopt};
        }
        else
        {
            auto s   = kumi::split(in.shape(), kumi::index<In::static_order -1>);
            auto ext = kumi::get<0>(s);                 
            auto inn = kumi::get<0>(kumi::get<1>(s));    
            auto pos = std::optional<coords_t>{std::nullopt};
            auto rmd = in.get_data();
            
            kwk::__::for_until([&](auto... is)
            {
                auto r_in = make_range(&in(is...,0), inn);
                auto find = eve::algo::find_if(r_in, f);
                if(find < r_in.end())
                {
                    auto linear_pos = find - rmd;
                    auto kwk_pos = kwk::coordinates(linear_pos, in.shape());
                    pos = kumi::to_tuple(kwk_pos);

                    return true;
                }
                return false;
            }, ext);

            return pos;
        }
    }

    //
    template<typename Context, concepts::container In, typename Func>
    constexpr auto find_if_not([[maybe_unused]] Context& ctx, In const& in, Func f)
    {
        return kwk::find_if("simd", in, [f](auto x){return !f(x);});
    }

    //
    template<typename Context, concepts::container In, concepts::container Values>
    constexpr auto find_first_of([[maybe_unused]] Context& ctx, In const& in, Values const& values)
    {
        return kwk::find_if("simd",in, [&values](auto e)
        {
            // Weird workaround, cannot return the result of "any_of" directly
            // due to EVE algos (kwk::find_if simd calls eve::find_if)
            if (kwk::any_of("simd", values, [&](auto x){return (x==e);}))
                return e == e;
            else
                return e != e;
        });
    }
    
    //
    template<typename Context, concepts::container In>
    constexpr auto find_last([[maybe_unused]] Context& ctx, In const& in, auto value)
    {
        return find_last_if("simd", in, [&](auto e){return e == value;});
    }

    //
    template<typename Context, concepts::container In, typename Func>
    constexpr auto find_last_if([[maybe_unused]] Context& ctx, In const& in, Func f)
    {
        auto c = kumi::generate<In::static_order, std::ptrdiff_t>(0);
        using coords_t = decltype(c);

        if constexpr (In::preserve_reachability)
        {
            auto r_in = make_range(in.get_data(), in.numel());
            auto find = eve::algo::find_last_if(r_in, f);
            if(find < r_in.end())
            {
                auto linear_pos  = find - r_in.begin();
                auto pos = kwk::coordinates(linear_pos, in.shape());
                return std::optional<coords_t>{kumi::to_tuple(pos)};
            }
            return std::optional<coords_t>{std::nullopt};
        }
        else
        {
            auto s   = kumi::split(in.shape(), kumi::index<In::static_order -1>);
            auto ext = kumi::get<0>(s);                 
            auto inn = kumi::get<0>(kumi::get<1>(s));    
            auto pos = std::optional<coords_t>{std::nullopt};
            auto rmd = in.get_data();

            kwk::__::for_until([&](auto... is)
            {
                auto r_in = make_range(&in(is...,0), inn);
                auto find = eve::algo::find_last_if(r_in, f);
                if(find < r_in.end())
                {
                    auto linear_pos = find - rmd;
                    auto kwk_pos = kwk::coordinates(linear_pos, in.shape());
                    pos = std::optional<coords_t>{kumi::to_tuple(kwk_pos)};

                    return true;
                }
                return false;
            }, ext);

            return pos;
        }
    }

    //
    template<typename Context, concepts::container In, typename Func>
    constexpr auto find_last_if_not([[maybe_unused]] Context& ctx, In const& in, Func f)
    {
        return kwk::find_last_if("simd", in, [f](auto x){return !f(x);});
    }

    /*************************** GENERATOR ******************************/
    //
    template<typename Context, concepts::container Inout>
    constexpr auto fill([[maybe_unused]] Context &ctx, Inout& inout, auto value)
    {
        if constexpr (Inout::preserve_reachability)
        {
            auto r_inout = make_range(inout.get_data(), inout.numel());
            eve::algo::fill(r_inout, value);
        }
        else
        {
            auto s   = kumi::split(inout.shape(), kumi::index<Inout::static_order -1>);
            auto ext = kumi::get<0>(s);                
            auto inn = kumi::get<0>(kumi::get<1>(s)); 

            kwk::__::for_each([&](auto... is)
            {
                auto r_inout = make_range(&inout(is...,0), inn);
                eve::algo::fill(r_inout, value);
            }, ext);
        }
    }

    template <typename Wide_t> 
    constexpr auto wide_to_coords(Wide_t const &wide, auto relative_pos)
    {
        auto span = eve::views::zip(eve::views::iota(relative_pos), wide);

    }
    
    /*
        Case generate on func with multi params

        Generate is actually a eve::tranform_to from a range of tuples
        containing the multi dim index of each value 

        Base case :

        Generate takes the current linear index of the container
        and apply a eve::tranform_inplace
    */
    // THIS IS NOT OKAY
    template<typename Context, typename Generator, concepts::container Inout>
    constexpr auto generate([[maybe_unused]] Context &ctx, Generator g, Inout& inout)
    {
       // if constexpr (Inout::preserve_reachability)
        //{
            auto r_inout = make_range(inout.get_data(), inout.numel());
            auto span = eve::views::zip(eve::views::iota(0), r_inout);

            //test(inout.shape());
            eve::algo::transform_to(span, r_inout,
                    [&](auto i)
                    {
                        return g(get<1>(i));
                    });
            //}
        /*else
        {
            auto s   = kumi::split(inout.shape(), kumi::index<Inout::static_order -1>);
            auto ext = kumi::get<0>(s);                
            auto inn = kumi::get<0>(kumi::get<1>(s)); 
            auto acc = 0;
            kwk::__::for_each([&](auto... is)
            {
                auto r_inout = make_range(&inout(is...,0), inn);
                eve::algo::transform_to(eve::views::iota(acc), r_inout,
                    [&](auto iota)
                    {
                        acc += inn;
                        return g(iota);
                    });

            }, ext);
        }*/
    }

    //
    template<typename Context, concepts::container Inout>
    constexpr auto iota ([[maybe_unused]] Context& ctx, Inout& inout, auto value)
    {
        if constexpr (Inout::preserve_reachability)
        {
            auto r_inout = make_range(inout.get_data(), inout.numel());
            eve::algo::iota(r_inout, value);
        }
        else
        {
            auto s   = kumi::split(inout.shape(), kumi::index<Inout::static_order -1>);
            auto ext = kumi::get<0>(s);                
            auto inn = kumi::get<0>(kumi::get<1>(s)); 
            auto acc = value; 
            kwk::__::for_each([&](auto... is)
            {
                auto r_inout = make_range(&inout(is...,0), inn);
                eve::algo::iota(r_inout, acc);
                acc += inn;
            }, ext);
        }
    }

    //
    template<typename Context, concepts::container Inout>
    constexpr auto iota ([[maybe_unused]] Context& ctx, Inout& inout, auto value, auto step)
    {
        if constexpr (Inout::preserve_reachability)
        {
            auto r_inout = make_range(inout.get_data(), inout.numel());
            eve::algo::transform_to(eve::views::iota_with_step(value, step), r_inout,
                    [&](auto iota)
                    {
                        return iota;
                    });
        }
        else
        {
            auto s   = kumi::split(inout.shape(), kumi::index<Inout::static_order -1>);
            auto ext = kumi::get<0>(s);                
            auto inn = kumi::get<0>(kumi::get<1>(s)); 
            auto acc = value;
            kwk::__::for_each([&](auto... is)
            {
                auto r_inout = make_range(&inout(is...,0), inn);
                eve::algo::transform_to(eve::views::iota_with_step(acc, step), r_inout,
                    [&](auto iota)
                    {
                        return iota;
                    });
                acc += step*inn;
            }, ext);
        }
    }
}
