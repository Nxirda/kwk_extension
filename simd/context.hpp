#pragma once 

#include <kwk/concepts/container.hpp>
#include <kwk/context/base.hpp>

namespace kwk
{
    struct simd_context : base_context<simd_context>
    {

    }

    inline constexpr simd_context simd = {};
}
