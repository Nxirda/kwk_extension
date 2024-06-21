#include <kumi/tuple.hpp>
#include <kwk/kwk.hpp>
#include <eve/eve.hpp>
#include <eve/module/algo.hpp>
namespace kwk
{

// Visibly cannot be a wide inside and we need to access a range which is not what we do 
    template<typename Context, typename Func, concepts::container Out
            ,concepts::container C0, concepts::container... Cs
            >
    constexpr auto transform(Context &ctx, Func &&f, Out& out, C0&& c0, Cs&&... cs)
    {
        auto r_in  = eve::algo::as_range(c0.get_data(), c0.get_data() + c0.numel());
        auto r_out = eve::algo::as_range(out.get_data(), out.get_data() + out.numel());
       
        auto zipped = eve::views::zip(r_in, r_in, r_in);
        eve::algo::transform_to(r_in, r_out, EVE_FWD(f));
    }
}
