#include "lcuda.h"

namespace lcuda
{
namespace
{
    inline LCUDA_HOST LCUDA_DEVICE u32 rotl(u32 x, s32 k)
    {
        return (x << k) | (x >> (32 - k));
    }

    // Return [0, 1)
    inline LCUDA_HOST LCUDA_DEVICE f32 toF32(u32 x)
    {
        static const u32 m0 = 0x3F800000U;
        static const u32 m1 = 0x007FFFFFU;
        x = m0|(x&m1);
        return (*(f32*)&x) - 1.000000000f;
    }

    LCUDA_HOST LCUDA_DEVICE u32 scramble(u32 x, u32 i)
    {
        return (1812433253 * (x^(x >> 30)) + i);
    }
}

    LCUDA_HOST LCUDA_DEVICE uint4 xoshiro128plus_srand(u32 seed)
    {
        u32 x0 = seed;
        u32 x1 = scramble(x0, 1);
        u32 x2 = scramble(x1, 2);
        u32 x3 = scramble(x2, 3);
        return ::make_uint4(x0, x1, x2, x3);
    }

    LCUDA_HOST LCUDA_DEVICE u32 xoshiro128plus_rand(uint4& r)
    {
        u32 result = r.x + r.z;
        u32 t = r.y << 9;

        r.z ^= r.x;
        r.w ^= r.y;
        r.y ^= r.z;
        r.x ^= r.w;

        r.z ^= t;
        r.w = rotl(r.w, 11);

        return result;
    }

    LCUDA_HOST LCUDA_DEVICE f32 xoshiro128plus_frand(uint4& r)
    {
        return toF32(xoshiro128plus_rand(r));
    }

    //----------------------------------------------------------------------------------
    LCUDA_HOST LCUDA_DEVICE float2 normalizeChecked(const float2& x, const float2& default)
    {
        f32 d = sqrtf(dot(x, x));
        if(d<=F32_EPSILON){
            return default;
        }
        f32 inv = 1.0f/d;
        return ::make_float2(x.x*inv, x.y*inv);
    }

    LCUDA_HOST LCUDA_DEVICE float3 normalizeChecked(const float3& x, const float3& default)
    {
        f32 d = sqrtf(dot(x, x));
        if(d<=F32_EPSILON){
            return default;
        }
        f32 inv = 1.0f/d;
        return ::make_float3(x.x*inv, x.y*inv, x.z*inv);
    }

    LCUDA_HOST LCUDA_DEVICE float4 normalizeChecked(const float4& x, const float4& default)
    {
        f32 d = sqrtf(dot(x, x));
        if(d<=F32_EPSILON){
            return default;
        }
        f32 inv = 1.0f/d;
        return ::make_float4(x.x*inv, x.y*inv, x.z*inv, x.w*inv);
    }

    //----------------------------------------------------------------------------------
    LCUDA_HOST LCUDA_DEVICE void orthonormalBasis(float3& binormal0, float3& binormal1, const float3& normal)
    {
        if(fabsf(normal.y)<fabsf(normal.x)){
            f32 invLen = 1.0f/sqrt(normal.x*normal.x + normal.z*normal.z);
            binormal1 = ::make_float3(normal.z*invLen, 0.0f, -normal.x*invLen);

        }else{
            f32 invLen = 1.0f/sqrt(normal.y*normal.y + normal.z*normal.z);
            binormal1 = ::make_float3(0.0f, normal.z*invLen, -normal.y*invLen);
        }
        binormal0 = cross(binormal1, normal);
    }

    LCUDA_HOST LCUDA_DEVICE void orthonormalBasis(float4& binormal0, float4& binormal1, const float4& normal)
    {
        if(fabsf(normal.y)<fabsf(normal.x)){
            f32 invLen = 1.0f/sqrt(normal.x*normal.x + normal.z*normal.z);
            binormal1 = ::make_float4(normal.z*invLen, 0.0f, -normal.x*invLen, 0.0f);

        }else{
            f32 invLen = 1.0f/sqrt(normal.y*normal.y + normal.z*normal.z);
            binormal1 = ::make_float4(0.0f, normal.z*invLen, -normal.y*invLen, 0.0f);
        }
        binormal0 = cross3(binormal1, normal);
    }
}
