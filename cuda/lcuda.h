#ifndef INC_LCUDA_H_
#define INC_LCUDA_H_
/**
@file lcuda.h
@author t-sakai
@date 2019/06/24
*/
#include <malloc.h>
#include <cstdint>
#include <cstddef>
#include <cassert>
#include <cmath>
#include <limits>
#include <type_traits>

#include <immintrin.h> //AVX intrinsics

#include <cuda.h>
#include <cuda_runtime.h>

#ifndef LCUDA_NULL
#   ifdef __cplusplus
#       if 201103L<=__cplusplus || 1900<=_MSC_VER
#           define LCUDA_CPP11 1
#       endif
#       ifdef LCUDA_CPP11
#           define LCUDA_NULL nullptr
#       else
#           define LCUDA_NULL 0
#       endif
#   else //__cplusplus
#       define LCUDA_NULL (void*)0
#   endif //__cplusplus
#endif

#ifdef _DEBUG
#define LASSERT(exp) assert(exp)
#else
#define LASSERT(exp)
#endif

#define LCUDA_GLOBAL __global__
#define LCUDA_HOST __host__
#define LCUDA_DEVICE __device__

#define LCUDA_CONSTANT __constant__

namespace lcuda
{
    typedef char Char;
    typedef wchar_t WChar;

    typedef int8_t s8;
    typedef int16_t s16;
    typedef int32_t s32;
    typedef int64_t s64;

    typedef uint8_t u8;
    typedef uint16_t u16;
    typedef uint32_t u32;
    typedef uint64_t u64;

    typedef float f32;
    typedef double f64;

    //typedef std::size_t size_t;
    typedef std::intptr_t intptr_t;
    typedef std::uintptr_t uintptr_t;
    typedef std::ptrdiff_t ptrdiff_t;

    using std::move;

    typedef __m256 lm256;
    typedef __m256i lm256i;
    typedef __m128 lm128;
    typedef __m128i lm128i;
    typedef __m64 lm64;

    static constexpr f32 F32_EPSILON = FLT_EPSILON;
    static constexpr f32 F64_EPSILON = DBL_EPSILON;

    static constexpr f32 F32_MAX = FLT_MAX;

    static constexpr f32 PI = static_cast<f32>(3.14159265358979323846);
    static constexpr f32 PI2 = static_cast<f32>(6.28318530717958647692);
    static constexpr f32 INV_PI = static_cast<f32>(0.31830988618379067153);
    static constexpr f32 INV_PI2 = static_cast<f32>(0.15915494309189533576);
    static constexpr f32 INV_PI4 = static_cast<f32>(0.07957747154594766788);
    static constexpr f32 PI_2 = static_cast<f32>(1.57079632679489661923);
    static constexpr f32 INV_PI_2 = static_cast<f32>(0.63661977236758134308);
    static constexpr f32 LOG2 = static_cast<f32>(0.693147180559945309417);
    static constexpr f32 INV_LOG2 = static_cast<f32>(1.0/0.693147180559945309417);

    static constexpr f64 PI_64 = static_cast<f64>(3.14159265358979323846);
    static constexpr f64 PI2_64 = static_cast<f64>(6.28318530717958647692);
    static constexpr f64 INV_PI_64 = static_cast<f64>(0.31830988618379067153);
    static constexpr f64 INV_PI2_64 = static_cast<f64>(0.15915494309189533576);
    static constexpr f64 PI_2_64 = static_cast<f64>(1.57079632679489661923);
    static constexpr f64 INV_PI_2_64 = static_cast<f64>(0.63661977236758134308);
    static constexpr f64 LOG2_64 = static_cast<f64>(0.693147180559945309417);
    static constexpr f64 INV_LOG2_64 = static_cast<f64>(1.0/0.693147180559945309417);

    static constexpr f32 DEG_TO_RAD = static_cast<f32>(1.57079632679489661923/90.0);
    static constexpr f32 RAD_TO_DEG = static_cast<f32>(90.0/1.57079632679489661923);

    static constexpr f32 ANGLE_LIMIT1 = (0.9999f);

    static constexpr f32 DOT_EPSILON = (1.0e-6f);
    static constexpr f32 ANGLE_EPSILON = (1.0e-6f);
    static constexpr f32 PDF_EPSILON = (1.0e-4f);
    static constexpr f32 RAY_EPSILON = (1.0e-4f);

    static const s32 InvalidID = -1;

    void printError(cudaError error, const Char* file, s32 line);
    void printError(cudaError error, const Char* func, const Char* file, s32 line);

#ifdef _DEBUG
#   define PRINTERROR(error, file, line) printError(error, file, line)
#   define PRINTERROR_FUNC(error, func, file, line) printError(error, func, file, line)
#else
#   define PRINTERROR(error, file, line)
#   define PRINTERROR_FUNC(error, func, file, line)
#endif

#define CHECK_RETURN_IF_FAILURE(status, result) if(cudaSuccess != status){ PRINTERROR(status, __FILE__, __LINE__); return result;}
#define CALL_RETURN_IF_FAILURE(func, result) {cudaError status = func;if(cudaSuccess != status){ PRINTERROR_FUNC(status, #func, __FILE__, __LINE__); return result;}}

    template<class T>
    T* lmalloc(s32 count)
    {
        LASSERT(0<=count);
        return reinterpret_cast<T*>(malloc(sizeof(T)*count));
    }

    template<class T>
    void lfree(T*& ptr)
    {
        free(ptr);
        ptr = LCUDA_NULL;
    }

    template<class T>
    T* cudaMalloc(s32 count)
    {
        LASSERT(0<=count);
        T* ptr = LCUDA_NULL;
        ::cudaMalloc(&ptr, sizeof(T)*count);
        return ptr;
    }

     template<class T>
    T* cudaMallocHost(s32 count)
    {
        LASSERT(0<=count);
        T* ptr = LCUDA_NULL;
        ::cudaMallocHost(&ptr, sizeof(T)*count);
        return ptr;
    }

    template<class T>
    T* cudaMallocManaged(s32 count)
    {
        LASSERT(0<=count);
        T* ptr = LCUDA_NULL;
        ::cudaMallocManaged(&ptr, sizeof(T)*count);
        return ptr;
    }

    template<class T>
    T* cudaMalloc2D(size_t& pitch, s32 width, s32 height)
    {
        LASSERT(0<=width);
        LASSERT(0<=height);

        T* ptr = LCUDA_NULL;
        pitch = sizeof(T)*width;
        ::cudaMallocPitch(&ptr, &pitch, width, height);
        return ptr;
    }

    template<class T>
    T* cudaMalloc3D(s32 width, s32 height)
    {
        LASSERT(0<=width);
        LASSERT(0<=height);

        T* ptr = LCUDA_NULL;
        size_t pitch = sizeof(T)*width;
        ::cudaMallocPitch(&ptr, &pitch, width, height);
        return ptr;
    }

    template<class T>
    void cudaFree(T*& ptr)
    {
        ::cudaFree(ptr);
        ptr = LCUDA_NULL;
    }

    template<class T>
    void cudaFreeHost(T*& ptr)
    {
        ::cudaFreeHost(ptr);
        ptr = LCUDA_NULL;
    }

    //----------------------------------------------------------------------------------
    LCUDA_HOST LCUDA_DEVICE uint4 xoshiro128plus_srand(lcuda::u32 seed);

    LCUDA_HOST LCUDA_DEVICE lcuda::u32 xoshiro128plus_rand(uint4& r);
    LCUDA_HOST LCUDA_DEVICE lcuda::f32 xoshiro128plus_frand(uint4& r);

    //----------------------------------------------------------------------------------
    class Ray
    {
    public:
        LCUDA_HOST LCUDA_DEVICE Ray(){}
        LCUDA_HOST LCUDA_DEVICE Ray(const float3& origin, const float3& direction)
            :origin_(origin)
            ,direction_(direction)
        {}

        float3 origin_;
        float3 direction_;
    };

    //----------------------------------------------------------------------------------
    inline LCUDA_HOST LCUDA_DEVICE bool isEqual(f32 x0, f32 x1, f32 epsilon=F32_EPSILON)
    {
        return fabs(x0-x1)<=epsilon;
    }

    //----------------------------------------------------------------------------------
    inline LCUDA_HOST LCUDA_DEVICE bool isZero(f32 x, f32 epsilon=F32_EPSILON)
    {
        return fabsf(x)<=epsilon;
    }

    inline LCUDA_HOST LCUDA_DEVICE bool isZero(const float2& x, f32 epsilon=F32_EPSILON)
    {
        return fabsf(x.x)<=epsilon
            && fabsf(x.y)<=epsilon;
    }

    inline LCUDA_HOST LCUDA_DEVICE bool isZero(const float3& x, f32 epsilon=F32_EPSILON)
    {
        return fabsf(x.x)<=epsilon
            && fabsf(x.y)<=epsilon
            && fabsf(x.z)<=epsilon;
    }

    inline LCUDA_HOST LCUDA_DEVICE bool isZero(const float4& x, f32 epsilon=F32_EPSILON)
    {
        return fabsf(x.x)<=epsilon
            && fabsf(x.y)<=epsilon
            && fabsf(x.z)<=epsilon
            && fabsf(x.w)<=epsilon;
    }

    //----------------------------------------------------------------------------------
    inline LCUDA_HOST LCUDA_DEVICE float2 make_float2(f32 x)
    {
        return ::make_float2(x, x);
    }

    inline LCUDA_HOST LCUDA_DEVICE int2 make_int2(s32 x)
    {
        return ::make_int2(x, x);
    }

    inline LCUDA_HOST LCUDA_DEVICE  uint2 make_uint2(u32 x)
    {
        return ::make_uint2(x, x);
    }

    inline LCUDA_HOST LCUDA_DEVICE float3 make_float3(f32 x)
    {
        return ::make_float3(x, x, x);
    }

    inline LCUDA_HOST LCUDA_DEVICE int3 make_int3(s32 x)
    {
        return ::make_int3(x, x, x);
    }

    inline LCUDA_HOST LCUDA_DEVICE  uint3 make_uint3(u32 x)
    {
        return ::make_uint3(x, x, x);
    }

    inline LCUDA_HOST LCUDA_DEVICE float4 make_float4(f32 x)
    {
        return ::make_float4(x, x, x, x);
    }

    inline LCUDA_HOST LCUDA_DEVICE int4 make_int4(s32 x)
    {
        return ::make_int4(x, x, x, x);
    }

    inline LCUDA_HOST LCUDA_DEVICE  uint4 make_uint4(u32 x)
    {
        return ::make_uint4(x, x, x, x);
    }

    //----------------------------------------------------------------------------------
    inline LCUDA_HOST LCUDA_DEVICE float2 operator-(const float2& x)
    {
        return ::make_float2(-x.x, -x.y);
    }

    inline LCUDA_HOST LCUDA_DEVICE int2 operator-(const int2& x)
    {
        return ::make_int2(-x.x, -x.y);
    }

    inline LCUDA_HOST LCUDA_DEVICE float3 operator-(const float3& x)
    {
        return ::make_float3(-x.x, -x.y, -x.z);
    }

    inline LCUDA_HOST LCUDA_DEVICE int3 operator-(const int3& x)
    {
        return ::make_int3(-x.x, -x.y, -x.z);
    }

    inline LCUDA_HOST LCUDA_DEVICE float4 operator-(const float4& x)
    {
        return ::make_float4(-x.x, -x.y, -x.z, -x.w);
    }

    inline LCUDA_HOST LCUDA_DEVICE int4 operator-(const int4& x)
    {
        return ::make_int4(-x.x, -x.y, -x.z, -x.w);
    }

    //----------------------------------------------------------------------------------
    inline LCUDA_HOST LCUDA_DEVICE float2 operator+(const float2& x0, const float2& x1)
    {
        return ::make_float2(x0.x+x1.x, x0.y+x1.y);
    }

    inline LCUDA_HOST LCUDA_DEVICE void operator+=(float2& x0, const float2& x1)
    {
        x0.x += x1.x;
        x0.y += x1.y;
    }

    inline LCUDA_HOST LCUDA_DEVICE int2 operator+(const int2& x0, const int2& x1)
    {
        return ::make_int2(x0.x+x1.x, x0.y+x1.y);
    }

    inline LCUDA_HOST LCUDA_DEVICE void operator+=(int2& x0, const int2& x1)
    {
        x0.x += x1.x;
        x0.y += x1.y;
    }

    inline LCUDA_HOST LCUDA_DEVICE uint2 operator+(const uint2& x0, const uint2& x1)
    {
        return ::make_uint2(x0.x+x1.x, x0.y+x1.y);
    }

    inline LCUDA_HOST LCUDA_DEVICE void operator+=(uint2& x0, const uint2& x1)
    {
        x0.x += x1.x;
        x0.y += x1.y;
    }

    inline LCUDA_HOST LCUDA_DEVICE float3 operator+(const float3& x0, const float3& x1)
    {
        return ::make_float3(x0.x+x1.x, x0.y+x1.y, x0.z+x1.z);
    }

    inline LCUDA_HOST LCUDA_DEVICE void operator+=(float3& x0, const float3& x1)
    {
        x0.x += x1.x;
        x0.y += x1.y;
        x0.z += x1.z;
    }

    inline LCUDA_HOST LCUDA_DEVICE int3 operator+(const int3& x0, const int3& x1)
    {
        return ::make_int3(x0.x+x1.x, x0.y+x1.y, x0.z+x1.z);
    }

    inline LCUDA_HOST LCUDA_DEVICE void operator+=(int3& x0, const int3& x1)
    {
        x0.x += x1.x;
        x0.y += x1.y;
        x0.z += x1.z;
    }

    inline LCUDA_HOST LCUDA_DEVICE uint3 operator+(const uint3& x0, const uint3& x1)
    {
        return ::make_uint3(x0.x+x1.x, x0.y+x1.y, x0.z+x1.z);
    }

    inline LCUDA_HOST LCUDA_DEVICE void operator+=(uint3& x0, const uint3& x1)
    {
        x0.x += x1.x;
        x0.y += x1.y;
        x0.z += x1.z;
    }

    inline LCUDA_HOST LCUDA_DEVICE float4 operator+(const float4& x0, const float4& x1)
    {
        return ::make_float4(x0.x+x1.x, x0.y+x1.y, x0.z+x1.z, x0.w+x1.w);
    }

    inline LCUDA_HOST LCUDA_DEVICE void operator+=(float4& x0, const float4& x1)
    {
        x0.x += x1.x;
        x0.y += x1.y;
        x0.z += x1.z;
        x0.w += x1.w;
    }

    inline LCUDA_HOST LCUDA_DEVICE int4 operator+(const int4& x0, const int4& x1)
    {
        return ::make_int4(x0.x+x1.x, x0.y+x1.y, x0.z+x1.z, x0.w+x1.w);
    }

    inline LCUDA_HOST LCUDA_DEVICE void operator+=(int4& x0, const int4& x1)
    {
        x0.x += x1.x;
        x0.y += x1.y;
        x0.z += x1.z;
        x0.w += x1.w;
    }

    inline LCUDA_HOST LCUDA_DEVICE uint4 operator+(const uint4& x0, const uint4& x1)
    {
        return ::make_uint4(x0.x+x1.x, x0.y+x1.y, x0.z+x1.z, x0.w+x1.w);
    }

    inline LCUDA_HOST LCUDA_DEVICE void operator+=(uint4& x0, const uint4& x1)
    {
        x0.x += x1.x;
        x0.y += x1.y;
        x0.z += x1.z;
        x0.w += x1.w;
    }

    //----------------------------------------------------------------------------------
    inline LCUDA_HOST LCUDA_DEVICE float2 operator-(const float2& x0, const float2& x1)
    {
        return ::make_float2(x0.x-x1.x, x0.y-x1.y);
    }

    inline LCUDA_HOST LCUDA_DEVICE void operator-=(float2& x0, const float2& x1)
    {
        x0.x -= x1.x;
        x0.y -= x1.y;
    }

    inline LCUDA_HOST LCUDA_DEVICE int2 operator-(const int2& x0, const int2& x1)
    {
        return ::make_int2(x0.x-x1.x, x0.y-x1.y);
    }

    inline LCUDA_HOST LCUDA_DEVICE void operator-=(int2& x0, const int2& x1)
    {
        x0.x -= x1.x;
        x0.y -= x1.y;
    }

    inline LCUDA_HOST LCUDA_DEVICE uint2 operator-(const uint2& x0, const uint2& x1)
    {
        return ::make_uint2(x0.x-x1.x, x0.y-x1.y);
    }

    inline LCUDA_HOST LCUDA_DEVICE void operator-=(uint2& x0, const uint2& x1)
    {
        x0.x -= x1.x;
        x0.y -= x1.y;
    }

    inline LCUDA_HOST LCUDA_DEVICE float3 operator-(const float3& x0, const float3& x1)
    {
        return ::make_float3(x0.x-x1.x, x0.y-x1.y, x0.z-x1.z);
    }

    inline LCUDA_HOST LCUDA_DEVICE void operator-=(float3& x0, const float3& x1)
    {
        x0.x -= x1.x;
        x0.y -= x1.y;
        x0.z -= x1.z;
    }

    inline LCUDA_HOST LCUDA_DEVICE int3 operator-(const int3& x0, const int3& x1)
    {
        return ::make_int3(x0.x-x1.x, x0.y-x1.y, x0.z-x1.z);
    }

    inline LCUDA_HOST LCUDA_DEVICE void operator-=(int3& x0, const int3& x1)
    {
        x0.x -= x1.x;
        x0.y -= x1.y;
        x0.z -= x1.z;
    }

    inline LCUDA_HOST LCUDA_DEVICE uint3 operator-(const uint3& x0, const uint3& x1)
    {
        return ::make_uint3(x0.x-x1.x, x0.y-x1.y, x0.z-x1.z);
    }

    inline LCUDA_HOST LCUDA_DEVICE void operator-=(uint3& x0, const uint3& x1)
    {
        x0.x -= x1.x;
        x0.y -= x1.y;
        x0.z -= x1.z;
    }

    inline LCUDA_HOST LCUDA_DEVICE float4 operator-(const float4& x0, const float4& x1)
    {
        return ::make_float4(x0.x-x1.x, x0.y-x1.y, x0.z-x1.z, x0.w-x1.w);
    }

    inline LCUDA_HOST LCUDA_DEVICE void operator-=(float4& x0, const float4& x1)
    {
        x0.x -= x1.x;
        x0.y -= x1.y;
        x0.z -= x1.z;
        x0.w -= x1.w;
    }

    inline LCUDA_HOST LCUDA_DEVICE int4 operator-(const int4& x0, const int4& x1)
    {
        return ::make_int4(x0.x-x1.x, x0.y-x1.y, x0.z-x1.z, x0.w-x1.w);
    }

    inline LCUDA_HOST LCUDA_DEVICE void operator-=(int4& x0, const int4& x1)
    {
        x0.x -= x1.x;
        x0.y -= x1.y;
        x0.z -= x1.z;
        x0.w -= x1.w;
    }

    inline LCUDA_HOST LCUDA_DEVICE uint4 operator-(const uint4& x0, const uint4& x1)
    {
        return ::make_uint4(x0.x-x1.x, x0.y-x1.y, x0.z-x1.z, x0.w-x1.w);
    }

    inline LCUDA_HOST LCUDA_DEVICE void operator-=(uint4& x0, const uint4& x1)
    {
        x0.x -= x1.x;
        x0.y -= x1.y;
        x0.z -= x1.z;
        x0.w -= x1.w;
    }

    //----------------------------------------------------------------------------------
    inline LCUDA_HOST LCUDA_DEVICE float2 operator*(f32 x0, const float2& x1)
    {
        return ::make_float2(x0*x1.x, x0*x1.y);
    }

    inline LCUDA_HOST LCUDA_DEVICE float2 operator*(const float2& x0, f32 x1)
    {
        return ::make_float2(x0.x*x1, x0.y*x1);
    }

    inline LCUDA_HOST LCUDA_DEVICE float2 operator*(const float2& x0, const float2& x1)
    {
        return ::make_float2(x0.x*x1.x, x0.y*x1.y);
    }

    inline LCUDA_HOST LCUDA_DEVICE void operator*=(float2& x0, f32 x1)
    {
        x0.x *= x1;
        x0.y *= x1;
    }

    inline LCUDA_HOST LCUDA_DEVICE void operator*=(float2& x0, const float2& x1)
    {
        x0.x *= x1.x;
        x0.y *= x1.y;
    }

    inline LCUDA_HOST LCUDA_DEVICE int2 operator*(const int2& x0, const int2& x1)
    {
        return ::make_int2(x0.x*x1.x, x0.y*x1.y);
    }

    inline LCUDA_HOST LCUDA_DEVICE void operator*=(int2& x0, const int2& x1)
    {
        x0.x *= x1.x;
        x0.y *= x1.y;
    }

    inline LCUDA_HOST LCUDA_DEVICE uint2 operator*(const uint2& x0, const uint2& x1)
    {
        return ::make_uint2(x0.x*x1.x, x0.y*x1.y);
    }

    inline LCUDA_HOST LCUDA_DEVICE void operator*=(uint2& x0, const uint2& x1)
    {
        x0.x *= x1.x;
        x0.y *= x1.y;
    }

    inline LCUDA_HOST LCUDA_DEVICE float3 operator*(f32 x0, const float3& x1)
    {
        return ::make_float3(x0*x1.x, x0*x1.y, x0*x1.z);
    }

    inline LCUDA_HOST LCUDA_DEVICE float3 operator*(const float3& x0, f32 x1)
    {
        return ::make_float3(x0.x*x1, x0.y*x1, x0.z*x1);
    }

    inline LCUDA_HOST LCUDA_DEVICE float3 operator*(const float3& x0, const float3& x1)
    {
        return ::make_float3(x0.x*x1.x, x0.y*x1.y, x0.z*x1.z);
    }

    inline LCUDA_HOST LCUDA_DEVICE void operator*=(float3& x0, f32 x1)
    {
        x0.x *= x1;
        x0.y *= x1;
        x0.z *= x1;
    }

    inline LCUDA_HOST LCUDA_DEVICE void operator*=(float3& x0, const float3& x1)
    {
        x0.x *= x1.x;
        x0.y *= x1.y;
        x0.z *= x1.z;
    }

    inline LCUDA_HOST LCUDA_DEVICE int3 operator*(const int3& x0, const int3& x1)
    {
        return ::make_int3(x0.x*x1.x, x0.y*x1.y, x0.z*x1.z);
    }

    inline LCUDA_HOST LCUDA_DEVICE void operator*=(int3& x0, const int3& x1)
    {
        x0.x *= x1.x;
        x0.y *= x1.y;
        x0.z *= x1.z;
    }

    inline LCUDA_HOST LCUDA_DEVICE uint3 operator*(const uint3& x0, const uint3& x1)
    {
        return ::make_uint3(x0.x*x1.x, x0.y*x1.y, x0.z*x1.z);
    }

    inline LCUDA_HOST LCUDA_DEVICE void operator*=(uint3& x0, const uint3& x1)
    {
        x0.x *= x1.x;
        x0.y *= x1.y;
        x0.z *= x1.z;
    }

    inline LCUDA_HOST LCUDA_DEVICE float4 operator*(f32 x0, const float4& x1)
    {
        return ::make_float4(x0*x1.x, x0*x1.y, x0*x1.z, x0*x1.w);
    }

    inline LCUDA_HOST LCUDA_DEVICE float4 operator*(const float4& x0, f32 x1)
    {
        return ::make_float4(x0.x*x1, x0.y*x1, x0.z*x1, x0.w*x1);
    }

    inline LCUDA_HOST LCUDA_DEVICE float4 operator*(const float4& x0, const float4& x1)
    {
        return ::make_float4(x0.x*x1.x, x0.y*x1.y, x0.z*x1.z, x0.w*x1.w);
    }

    inline LCUDA_HOST LCUDA_DEVICE void operator*=(float4& x0, f32 x1)
    {
        x0.x *= x1;
        x0.y *= x1;
        x0.z *= x1;
        x0.w *= x1;
    }

    inline LCUDA_HOST LCUDA_DEVICE void operator*=(float4& x0, const float4& x1)
    {
        x0.x *= x1.x;
        x0.y *= x1.y;
        x0.z *= x1.z;
        x0.w *= x1.w;
    }

    inline LCUDA_HOST LCUDA_DEVICE int4 operator*(const int4& x0, const int4& x1)
    {
        return ::make_int4(x0.x*x1.x, x0.y*x1.y, x0.z*x1.z, x0.w*x1.w);
    }

    inline LCUDA_HOST LCUDA_DEVICE void operator*=(int4& x0, const int4& x1)
    {
        x0.x *= x1.x;
        x0.y *= x1.y;
        x0.z *= x1.z;
        x0.w *= x1.w;
    }

    inline LCUDA_HOST LCUDA_DEVICE uint4 operator*(const uint4& x0, const uint4& x1)
    {
        return ::make_uint4(x0.x*x1.x, x0.y*x1.y, x0.z*x1.z, x0.w*x1.w);
    }

    inline LCUDA_HOST LCUDA_DEVICE void operator*=(uint4& x0, const uint4& x1)
    {
        x0.x *= x1.x;
        x0.y *= x1.y;
        x0.z *= x1.z;
        x0.w *= x1.w;
    }

    //----------------------------------------------------------------------------------
    inline LCUDA_HOST LCUDA_DEVICE float2 operator/(const float2& x0, f32 x1)
    {
        return ::make_float2(x0.x/x1, x0.y/x1);
    }

    inline LCUDA_HOST LCUDA_DEVICE float2 operator/(const float2& x0, const float2& x1)
    {
        return ::make_float2(x0.x/x1.x, x0.y/x1.y);
    }

    inline LCUDA_HOST LCUDA_DEVICE void operator/=(float2& x0, f32 x1)
    {
        x0.x /= x1;
        x0.y /= x1;
    }

    inline LCUDA_HOST LCUDA_DEVICE void operator/=(float2& x0, const float2& x1)
    {
        x0.x /= x1.x;
        x0.y /= x1.y;
    }

    inline LCUDA_HOST LCUDA_DEVICE int2 operator/(const int2& x0, const int2& x1)
    {
        return ::make_int2(x0.x/x1.x, x0.y/x1.y);
    }

    inline LCUDA_HOST LCUDA_DEVICE void operator/=(int2& x0, const int2& x1)
    {
        x0.x /= x1.x;
        x0.y /= x1.y;
    }

    inline LCUDA_HOST LCUDA_DEVICE uint2 operator/(const uint2& x0, const uint2& x1)
    {
        return ::make_uint2(x0.x/x1.x, x0.y/x1.y);
    }

    inline LCUDA_HOST LCUDA_DEVICE void operator/=(uint2& x0, const uint2& x1)
    {
        x0.x /= x1.x;
        x0.y /= x1.y;
    }

    inline LCUDA_HOST LCUDA_DEVICE float3 operator/(const float3& x0, f32 x1)
    {
        f32 inv = 1.0f/x1;
        return ::make_float3(x0.x*inv, x0.y*inv, x0.z*inv);
    }


    inline LCUDA_HOST LCUDA_DEVICE float3 operator/(const float3& x0, const float3& x1)
    {
        return ::make_float3(x0.x/x1.x, x0.y/x1.y, x0.z/x1.z);
    }

    inline LCUDA_HOST LCUDA_DEVICE void operator/=(float3& x0, f32 x1)
    {
        f32 inv = 1.0f/x1;
        x0.x *= inv;
        x0.y *= inv;
        x0.z *= inv;
    }

    inline LCUDA_HOST LCUDA_DEVICE void operator/=(float3& x0, const float3& x1)
    {
        x0.x /= x1.x;
        x0.y /= x1.y;
        x0.z /= x1.z;
    }

    inline LCUDA_HOST LCUDA_DEVICE int3 operator/(const int3& x0, const int3& x1)
    {
        return ::make_int3(x0.x/x1.x, x0.y/x1.y, x0.z/x1.z);
    }

    inline LCUDA_HOST LCUDA_DEVICE void operator/=(int3& x0, const int3& x1)
    {
        x0.x /= x1.x;
        x0.y /= x1.y;
        x0.z /= x1.z;
    }

    inline LCUDA_HOST LCUDA_DEVICE uint3 operator/(const uint3& x0, const uint3& x1)
    {
        return ::make_uint3(x0.x/x1.x, x0.y/x1.y, x0.z/x1.z);
    }

    inline LCUDA_HOST LCUDA_DEVICE void operator/=(uint3& x0, const uint3& x1)
    {
        x0.x /= x1.x;
        x0.y /= x1.y;
        x0.z /= x1.z;
    }

    inline LCUDA_HOST LCUDA_DEVICE float4 operator/(const float4& x0, f32 x1)
    {
        f32 inv = 1.0f/x1;
        return ::make_float4(x0.x*inv, x0.y*inv, x0.z*inv, x0.w*inv);
    }

    inline LCUDA_HOST LCUDA_DEVICE float4 operator/(const float4& x0, const float4& x1)
    {
        return ::make_float4(x0.x/x1.x, x0.y/x1.y, x0.z/x1.z, x0.w/x1.w);
    }

    inline LCUDA_HOST LCUDA_DEVICE void operator/=(float4& x0, f32 x1)
    {
        f32 inv = 1.0f/x1;
        x0.x *= inv;
        x0.y *= inv;
        x0.z *= inv;
        x0.w *= inv;
    }

    inline LCUDA_HOST LCUDA_DEVICE void operator/=(float4& x0, const float4& x1)
    {
        x0.x /= x1.x;
        x0.y /= x1.y;
        x0.z /= x1.z;
        x0.w /= x1.w;
    }

    inline LCUDA_HOST LCUDA_DEVICE int4 operator/(const int4& x0, const int4& x1)
    {
        return ::make_int4(x0.x/x1.x, x0.y/x1.y, x0.z/x1.z, x0.w/x1.w);
    }

    inline LCUDA_HOST LCUDA_DEVICE void operator/=(int4& x0, const int4& x1)
    {
        x0.x /= x1.x;
        x0.y /= x1.y;
        x0.z /= x1.z;
        x0.w /= x1.w;
    }

    inline LCUDA_HOST LCUDA_DEVICE uint4 operator/(const uint4& x0, const uint4& x1)
    {
        return ::make_uint4(x0.x/x1.x, x0.y/x1.y, x0.z/x1.z, x0.w/x1.w);
    }

    inline LCUDA_HOST LCUDA_DEVICE void operator/=(uint4& x0, const uint4& x1)
    {
        x0.x /= x1.x;
        x0.y /= x1.y;
        x0.z /= x1.z;
        x0.w /= x1.w;
    }

    //----------------------------------------------------------------------------------
    inline LCUDA_HOST LCUDA_DEVICE f32 lerp(f32 x0, f32 x1, f32 t)
    {
        return x0 + t*(x1-x0);
    }

    inline LCUDA_HOST LCUDA_DEVICE float2 lerp(const float2& x0, const float2& x1, const float2& t)
    {
        return x0 + t*(x1-x0);
    }

    inline LCUDA_HOST LCUDA_DEVICE float3 lerp(const float3& x0, const float3& x1, const float3& t)
    {
        return x0 + t*(x1-x0);
    }

    inline LCUDA_HOST LCUDA_DEVICE float4 lerp(const float4& x0, const float4& x1, const float4& t)
    {
        return x0 + t*(x1-x0);
    }

    //----------------------------------------------------------------------------------
    inline LCUDA_HOST LCUDA_DEVICE f32 dot(const float2& x0, const float2& x1)
    {
        return x0.x*x1.x + x0.y*x1.y;
    }

    inline LCUDA_HOST LCUDA_DEVICE f32 dot(const float3& x0, const float3& x1)
    {
        return x0.x*x1.x + x0.y*x1.y + x0.z*x1.z;
    }

    inline LCUDA_HOST LCUDA_DEVICE f32 dot(const float4& x0, const float4& x1)
    {
        return x0.x*x1.x + x0.y*x1.y + x0.z*x1.z + x0.w*x1.w;
    }

    //----------------------------------------------------------------------------------
    inline LCUDA_HOST LCUDA_DEVICE f32 length(const float2& x)
    {
        return sqrtf(dot(x, x));
    }

    inline LCUDA_HOST LCUDA_DEVICE f32 length(const float3& x)
    {
        return sqrtf(dot(x, x));
    }

    inline LCUDA_HOST LCUDA_DEVICE f32 length(const float4& x)
    {
        return sqrtf(dot(x, x));
    }

    //----------------------------------------------------------------------------------
    inline LCUDA_HOST LCUDA_DEVICE float2 normalize(const float2& x)
    {
        f32 inv = 1.0f/sqrtf(dot(x, x));
        return ::make_float2(x.x*inv, x.y*inv);
    }

    inline LCUDA_HOST LCUDA_DEVICE float3 normalize(const float3& x)
    {
        f32 inv = 1.0f/sqrtf(dot(x, x));
        return ::make_float3(x.x*inv, x.y*inv, x.z*inv);
    }

    inline LCUDA_HOST LCUDA_DEVICE float4 normalize(const float4& x)
    {
        f32 inv = 1.0f/sqrtf(dot(x, x));
        return ::make_float4(x.x*inv, x.y*inv, x.z*inv, x.w*inv);
    }

    //----------------------------------------------------------------------------------
    LCUDA_HOST LCUDA_DEVICE float2 normalizeChecked(const float2& x, const float2& default);

    LCUDA_HOST LCUDA_DEVICE float3 normalizeChecked(const float3& x, const float3& default);

    LCUDA_HOST LCUDA_DEVICE float4 normalizeChecked(const float4& x, const float4& default);

    //----------------------------------------------------------------------------------
    inline LCUDA_HOST LCUDA_DEVICE float3 cross(const float3& x0, const float3& x1)
    {
        return ::make_float3(x0.y*x1.z - x0.z*x1.y, x0.z*x1.x - x0.x*x1.z, x0.x*x1.y - x0.y*x1.x);
    }

    inline LCUDA_HOST LCUDA_DEVICE float4 cross3(const float4& x0, const float4& x1)
    {
        return ::make_float4(x0.y*x1.z - x0.z*x1.y, x0.z*x1.x - x0.x*x1.z, x0.x*x1.y - x0.y*x1.x, 0.0f);
    }

    //----------------------------------------------------------------------------------
    inline LCUDA_HOST LCUDA_DEVICE float2 abs(const float2& x)
    {
        return ::make_float2(::abs(x.x), ::abs(x.y));
    }

    inline LCUDA_HOST LCUDA_DEVICE float3 abs(const float3& x)
    {
        return ::make_float3(::abs(x.x), ::abs(x.y), ::abs(x.z));
    }

    inline LCUDA_HOST LCUDA_DEVICE float4 abs(const float4& x)
    {
        return ::make_float4(::abs(x.x), ::abs(x.y), ::abs(x.z), ::abs(x.w));
    }

    //----------------------------------------------------------------------------------
    inline LCUDA_HOST LCUDA_DEVICE float2 min(const float2& x0, const float2& x1)
    {
        return ::make_float2(::fminf(x0.x, x1.x), ::fminf(x0.y, x1.y));
    }

    inline LCUDA_HOST LCUDA_DEVICE float3 min(const float3& x0, const float3& x1)
    {
        return ::make_float3(::fminf(x0.x, x1.x), ::fminf(x0.y, x1.y), ::fminf(x0.z, x1.z));
    }

    inline LCUDA_HOST LCUDA_DEVICE float4 min(const float4& x0, const float4& x1)
    {
        return ::make_float4(::fminf(x0.x, x1.x), ::fminf(x0.y, x1.y), ::fminf(x0.z, x1.z), ::fminf(x0.w, x1.w));
    }

    //----------------------------------------------------------------------------------
    inline LCUDA_HOST LCUDA_DEVICE float2 max(const float2& x0, const float2& x1)
    {
        return ::make_float2(::fmaxf(x0.x, x1.x), ::fmaxf(x0.y, x1.y));
    }

    inline LCUDA_HOST LCUDA_DEVICE float3 max(const float3& x0, const float3& x1)
    {
        return ::make_float3(::fmaxf(x0.x, x1.x), ::fmaxf(x0.y, x1.y), ::fmaxf(x0.z, x1.z));
    }

    inline LCUDA_HOST LCUDA_DEVICE float4 max(const float4& x0, const float4& x1)
    {
        return ::make_float4(::fmaxf(x0.x, x1.x), ::fmaxf(x0.y, x1.y), ::fmaxf(x0.z, x1.z), ::fmaxf(x0.w, x1.w));
    }

    //----------------------------------------------------------------------------------
    inline LCUDA_HOST LCUDA_DEVICE f32 clamp(f32 x, f32 minx, f32 maxx)
    {
        return ::fmaxf(minx, ::fminf(x, maxx));
    }

    inline LCUDA_HOST LCUDA_DEVICE float2 clamp(const float2& x, f32 minx, f32 maxx)
    {
        return ::make_float2(clamp(x.x, minx, maxx), clamp(x.y, minx, maxx));
    }

    inline LCUDA_HOST LCUDA_DEVICE float3 clamp(const float3& x, f32 minx, f32 maxx)
    {
        return ::make_float3(clamp(x.x, minx, maxx), clamp(x.y, minx, maxx), clamp(x.z, minx, maxx));
    }

    inline LCUDA_HOST LCUDA_DEVICE float4 clamp(const float4& x, f32 minx, f32 maxx)
    {
        return ::make_float4(clamp(x.x, minx, maxx), clamp(x.y, minx, maxx), clamp(x.z, minx, maxx), clamp(x.w, minx, maxx));
    }

    //----------------------------------------------------------------------------------
    inline LCUDA_HOST LCUDA_DEVICE f32 clamp01(f32 x)
    {
        return ::fmaxf(0.0f, ::fminf(x, 1.0f));
    }

    inline LCUDA_HOST LCUDA_DEVICE float2 clamp01(const float2& x)
    {
        return ::make_float2(clamp01(x.x), clamp01(x.y));
    }

    inline LCUDA_HOST LCUDA_DEVICE float3 clamp01(const float3& x)
    {
        return ::make_float3(clamp01(x.x), clamp01(x.y), clamp01(x.z));
    }

    inline LCUDA_HOST LCUDA_DEVICE float4 clamp01(const float4& x)
    {
        return ::make_float4(clamp01(x.x), clamp01(x.y), clamp01(x.z), clamp01(x.w));
    }

    //----------------------------------------------------------------------------------
    LCUDA_HOST LCUDA_DEVICE void orthonormalBasis(float3& binormal0, float3& binormal1, const float3& normal);
    LCUDA_HOST LCUDA_DEVICE void orthonormalBasis(float4& binormal0, float4& binormal1, const float4& normal);

    //----------------------------------------------------------------------------------
    /**
    From helper_cuda.h
    */
    s32 getSMVer2Cores(s32 major, s32 minor);
    s32 getMaximumPerformanceDevice();
    s32 initializeDevice(s32 deviceId);
}
#endif //INC_LCUDA_H_
