#include <kumi/tuple.hpp>
#include <kwk/kwk.hpp>
#include <eve/eve.hpp>
#include <eve/module/algo.hpp> 

namespace kwk
{

    //Simple function that builds a range on a kwk container
    template <concepts::container In>
    auto make_range(In &c)
    {
        return eve::algo::as_range(c.get_data(), c.get_data() + c.numel());
    }

    /********************* Transform  *********************************/
    // Context signature isnt correct ofc (simplified)
    template<typename Context, typename Func, concepts::container Out
            ,concepts::container C0, concepts::container... Cs
            >
    constexpr auto transform(Context &ctx, Func &&f, Out& out, C0&& c0, Cs&&... cs)
    {
        auto r_out = make_range(out);
       
        auto zipped = eve::views::zip(make_range(c0), make_range(cs)...);

        eve::algo::transform_to(zipped, r_out, [&f](auto const &in){return kumi::apply(f, in);});
    }
   
    /********************* Reductions *********************************/
    //We need the same semantic as eve::reduce ?
    //Also : that shall be in an inner loop (or on the flattenned array)
    template<typename Context, concepts::container In>
    constexpr auto reduce(Context &ctx, In const& in)
    {
        auto r_in = make_range(in);
        return eve::algo::reduce(r_in, typename In::value_type{});
    }
        
    /* 
     * This is fully temporary as only the last called algo shall
     * be vectorized ppbly (as algos are calling algos etc)
     */

    // This one idk yet, transform_reduce is not practical
    template<typename Context, typename Func, concepts::container In>
    constexpr auto reduce(Context& ctx, In const& in, Func f, auto init)
    {
        auto r_in = make_range(in);
        // Might need to have smth for out ?
        //kwk::transform("simd", f, r_in, r_in);

        return 0;//eve::algo::transform_reduce(r_in, f, init);
    }



    /********************* Copy *********************************/
    //
    template<typename Context, concepts::container Out, concepts::container In>
    constexpr auto copy(Context& ctx, Out& out, In&& in)
    {
        auto r_in  = make_range(in);
        auto r_out = make_range(out);
        eve::algo::copy(r_in, r_out);
    }

    //
    template<typename Context, typename Func,
             concepts::container Out, concepts::container In
            >
    constexpr auto copy_if(Context& ctx, Func f, Out& out, In&& in)
    {
        auto r_in  = make_range(in);
        auto r_out = make_range(out);
        eve::algo::copy_if(r_in, r_out, f);
    }

    /********************* Predicates *********************************/
    //
    template<typename Context, typename Func,
             concepts::container In
            >
    constexpr auto all_of(Context& ctx, In const& in, Func f)
    {
        auto r_in = make_range(in);
        return eve::algo::all_of(r_in, f);
    }

    //
    template<typename Context, typename Func,
             concepts::container In
            >
    constexpr auto any_of(Context& ctx, In const& in, Func f)
    {
        auto r_in = make_range(in);
        return eve::algo::any_of(r_in, f);
    }

    //
    template<typename Context, typename Func,
             concepts::container In
            >
    constexpr auto none_of(Context& ctx, In const& in, Func f)
    {
        auto r_in = make_range(in);
        return eve::algo::none_of(r_in, f);
    }

    /********************* Finds *********************************/
    //
    template<typename Context, concepts::container In>
    constexpr auto find(Context& ctx, In const& in, auto value)
    {
        auto r_in = make_range(in);
        return eve::algo::find(r_in, value);
    }

    //
    template<typename Context, concepts::container In, typename Func>
    constexpr auto find_if(Context& ctx, In const& in, Func f)
    {
        auto r_in = make_range(in);
        return eve::algo::find_if(r_in, f);
    }

    //
    template<typename Context, concepts::container In, typename Func>
    constexpr auto find_if_not(Context& ctx, In const& in, Func f)
    {
        auto r_in = make_range(in);
        return eve::algo::find_if_not(r_in, f);
    }

    // Have to come back to this
    template<typename Context, concepts::container In, concepts::container Values>
    constexpr auto find_first_of(Context& ctx, In const& in, Values const& values)
    {
        return kwk::find_if(in, [&](auto e)
        {
            return kwk::any_of("simd", values, [&](auto x){return (x==e);});
        });
    }
    
    //
    template<typename Context, concepts::container In>
    constexpr auto find_last(Context& ctx, In const& in, auto value)
    {
        auto r_in = make_range(in);
        return eve::algo::find_last(r_in, value);
    }

    //
    template<typename Context, concepts::container In, typename Func>
    constexpr auto find_last_if(Context& ctx, In const& in, Func f)
    {
        auto r_in = make_range(in);
        return eve::algo::find_last_if(r_in, f);
    }

    //
    template<typename Context, concepts::container In, typename Func>
    constexpr auto find_last_if_not(Context& ctx, In const& in, Func f)
    {
        auto r_in = make_range(in);
        return eve::algo::find_last_if_not(r_in, f);
    }
}
