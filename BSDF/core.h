#ifndef INC_LRENDER_H_
#define INC_LRENDER_H_
/**
@file core.h
@author t-sakai
@date 2019/02/13
*/
#include <malloc.h>
#include <cstdint>
#include <cstddef>
#include <cassert>
#include <cmath>
#include <limits>
#include <type_traits>
#include <string>

#include <immintrin.h> //AVX intrinsics

#ifndef LRENDER_NULL
#   ifdef __cplusplus
#       if 201103L<=__cplusplus || 1900<=_MSC_VER
#           define LRENDER_CPP11 1
#       endif
#       ifdef LRENDER_CPP11
#           define LRENDER_NULL nullptr
#       else
#           define LRENDER_NULL 0
#       endif
#   else //__cplusplus
#       define LRENDER_NULL (void*)0
#   endif //__cplusplus
#endif

#ifdef _DEBUG
#define LASSERT(exp) assert(exp)
#else
#define LASSERT(exp)
#endif

namespace lrender
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

    typedef std::size_t size_t;
    typedef std::intptr_t intptr_t;
    typedef std::uintptr_t uintptr_t;
    typedef std::ptrdiff_t ptrdiff_t;

    using std::move;

    typedef __m256 lm256;
    typedef __m256i lm256i;
    typedef __m128 lm128;
    typedef __m128i lm128i;
    typedef __m64 lm64;

#if defined(ANDROID)
    static constexpr f32 F32_EPSILON = 1.192092896e-07F;
    static constexpr f32 F64_EPSILON = 2.2204460492503131e-016;
#else
    static constexpr f32 F32_EPSILON = FLT_EPSILON;
    static constexpr f32 F64_EPSILON = DBL_EPSILON;
#endif

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

    static const Char CharNull = '\0';
    static const Char CharLF = '\n'; //Line Feed
    static const Char CharCR = '\r'; //Carriage Return
    static const Char PathDelimiter = '/';
    static const Char PathDelimiterWin = '\\';

#if defined(ANDROID) || defined(__GNUC__)
    typedef clock_t ClockType;
#else
    typedef u64 ClockType;
#endif

    //------------------------------------------------------
    //---
    //--- Utility functions
    //---
    //------------------------------------------------------
#define GFX_NEW new
#define GFX_PLACEMENT_NEW(ptr) new(ptr)
#define GFX_DELETE(ptr) delete ptr; (ptr)=LRENDER_NULL
#define GFX_DELETE_ARRAY(ptr) delete[] ptr; (ptr)=LRENDER_NULL
#define GFX_MALLOC(size) ::malloc(size)
#define GFX_FREE(ptr) ::free(ptr); (ptr)=LRENDER_NULL

#ifdef _MSC_VER
#define GFX_ALIGN16 __declspec(align(16))
#define GFX_ALIGN32 __declspec(align(32))
#define GFX_ALIGN(a) __declspec(align(a))
#define GFX_ALIGN_VAR16(type,x) __declspec(align(16)) type x
#define GFX_ALIGN_VAR(a,type,x) __declspec(align(a)) type x
#else
#define GFX_ALIGN16 __attribute__((aligned(16)))
#define GFX_ALIGN32 __attribute__((aligned(32)))
#define GFX_ALIGN(a) __attribute__((aligned(a)))
#define GFX_ALIGN_VAR16(type,x) type x __attribute__((aligned(16)))
#define GFX_ALIGN_VAR(a,type,x) type x __attribute__((aligned(a)))
#endif

#ifdef _DEBUG
#define DEBUG_ENABLE (true)
#else
#define DEBUG_ENABLE (false)
#endif

    static const uintptr_t GFX_ALIGN16_MASK = (0xFU);

#define LRENDER_ANGLE_LIMIT1 (0.9999f)

#define LRENDER_DOT_EPSILON (1.0e-6f)
#define LRENDER_ANGLE_EPSILON (1.0e-6f)
#define LRENDER_PDF_EPSILON (1.0e-4f)
#define LRENDER_RAY_EPSILON (1.0e-4f)

#define GFX_INFINITY numeric_limits<f32>::maximum()

    extern GFX_ALIGN32 const f32 One[8];

    union UnionS32F32
    {
        s32 s32_;
        f32 f32_;
    };

    union UnionU32F32
    {
        u32 u32_;
        f32 f32_;
    };

    union UnionS64F64
    {
        s64 s64_;
        f64 f64_;
    };

    union UnionU64F64
    {
        u64 u64_;
        f64 f64_;
    };

    template<class T>
    inline T* align16(T* ptr)
    {
        return (T*)(((uintptr_t)(ptr)+GFX_ALIGN16_MASK) & ~GFX_ALIGN16_MASK);
    }

    template<class T>
    T lerp(const T& v0, const T& v1, f32 ratio)
    {
        return static_cast<T>(v0 + ratio*(v1-v0));
    }

    template<class T>
    void swap(T& x0, T& x1)
    {
        T tmp = x0;
        x1 = x0;
        x0 = tmp;
    }

    template<class T>
    T minimum(const T& x0, const T& x1)
    {
        return x0<x1? x0: x1;
    }

    template<class T>
    T maximum(const T& x0, const T& x1)
    {
        return x0<x1? x1: x0;
    }

    template<class T>
    inline T absolute(T val)
    {
        return abs(val);
    }


    template<>
    inline u8 absolute<u8>(u8 val)
    {
        return val;
    }

    template<>
    inline u16 absolute<u16>(u16 val)
    {
        return val;
    }

    template<>
    inline u32 absolute<u32>(u32 val)
    {
        return val;
    }

    template<>
    f32 absolute<f32>(f32 val);

    template<>
    inline f64 absolute<f64>(f64 val)
    {
        return fabs(val);
    }

    template<class T>
    T clamp(const T x, const T minx, const T maxx)
    {
        return (x<=maxx)? ((minx<=x)? x : minx) : maxx;
    }

#ifdef _DEBUG
    void log(const Char* format, ...);
#define LOG(format, ...) log((format), __VA_ARGS__)
#else
#define LOG(format, ...)
#endif

    f32 clamp01(f32 x);

#define GFX_NONCOPYABLE(NAME)\
    NAME(const NAME&) =delete;\
    NAME(NAME&&) =delete;\
    NAME& operator=(const NAME&) =delete;\
    NAME& operator=(NAME&&) =delete

    inline bool isEqual(f32 x1, f32 x2)
    {
        return (absolute<f32>(x1 - x2) <= F32_EPSILON);
    }

    inline bool isEqual(f32 x1, f32 x2, f32 epsilon)
    {
        return (absolute<f32>(x1 - x2) <= epsilon);
    }

    inline bool isEqual(f64 x1, f64 x2)
    {
        return (absolute<f64>(x1 - x2) <= F64_EPSILON);
    }

    inline bool isEqual(f64 x1, f64 x2, f64 epsilon)
    {
        return (absolute<f64>(x1 - x2) <= epsilon);
    }

    inline bool isZero(f32 x1)
    {
        return (absolute<f32>(x1) <= F32_EPSILON);
    }

    inline bool isZero(f32 x1, f32 epsilon)
    {
        return (absolute<f32>(x1) <= epsilon);
    }

    inline bool isZero(f64 x1)
    {
        return (absolute<f64>(x1) <= F64_EPSILON);
    }

    inline bool isZero(f64 x1, f64 epsilon)
    {
        return (absolute<f64>(x1) <= epsilon);
    }

    inline bool isZeroPositive(f32 x1)
    {
        LASSERT(0.0f<=x1);
        return (x1 <= F32_EPSILON);
    }

    inline bool isZeroPositive(f32 x1, f32 epsilon)
    {
        LASSERT(0.0f<=x1);
        return (x1 <= epsilon);
    }

    inline bool isZeroPositive(f64 x1)
    {
        LASSERT(0.0f<=x1);
        return (x1 <= F64_EPSILON);
    }

    inline bool isZeroPositive(f64 x1, f64 epsilon)
    {
        LASSERT(0.0f<=x1);
        return (x1 <= epsilon);
    }

    inline bool isZeroNegative(f32 x1)
    {
        LASSERT(x1<=0.0f);
        return (-F32_EPSILON<=x1);
    }

    inline bool isZeroNegative(f32 x1, f32 epsilon)
    {
        LASSERT(x1<0.0f);
        return (epsilon<=x1);
    }

    inline bool isZeroNegative(f64 x1)
    {
        LASSERT(x1<0.0f);
        return (-F64_EPSILON<=x1);
    }

    inline bool isZeroNegative(f64 x1, f64 epsilon)
    {
        LASSERT(x1<0.0f);
        return (epsilon<=x1);
    }

    inline bool isNan(f32 f)
    {
        return ::isnan(f);
    }

    inline bool isNan(f64 f)
    {
        return ::isnan(f);
    }

    f32 sqrt(f32 x);

    inline f64 sqrt(f64 x)
    {
        return ::sqrt(x);
    }

    inline f32 sinf(f32 x)
    {
        return ::sinf(x);
    }

    inline f64 sin(f64 x)
    {
        return ::sin(x);
    }

    inline f32 cosf(f32 x)
    {
        return ::cosf(x);
    }

    inline f64 cos(f64 x)
    {
        return ::cos(x);
    }

    inline f32 acos(f32 x)
    {
        return ::acosf(x);
    }

    inline f64 acos(f64 x)
    {
        return ::acos(x);
    }

    inline f32 asin(f32 x)
    {
        return ::asinf(x);
    }

    inline f64 asin(f64 x)
    {
        return ::asin(x);
    }

    inline f32 atan(f32 x)
    {
        return ::atanf(x);
    }

    inline f64 atan(f64 x)
    {
        return ::atan(x);
    }

    inline f32 atan2(f32 x, f32 y)
    {
        return ::atan2f(x, y);
    }

    inline f64 atan2(f64 x, f64 y)
    {
        return ::atan2(x, y);
    }

    inline f32 exp(f32 x)
    {
        return ::expf(x);
    }

    inline f64 exp(f64 x)
    {
        return ::exp(x);
    }

    inline f32 exp2(f32 x)
    {
        return ::exp2f(x);
    }

    inline f64 exp2(f64 x)
    {
        return ::exp2(x);
    }

    inline f32 log(f32 x)
    {
        return ::logf(x);
    }

    inline f64 log(f64 x)
    {
        return ::log(x);
    }

    inline f32 log2(f32 x)
    {
        return ::log2f(x);
    }

    inline f64 log2(f64 x)
    {
        return ::log2(x);
    }

    inline f32 pow(f32 x, f32 y)
    {
        return ::powf(x, y);
    }

    inline f64 pow(f64 x, f64 y)
    {
        return ::pow(x, y);
    }

    inline f32 floor(f32 val)
    {
        return floorf(val);
    }

    inline f32 ceil(f32 val)
    {
        return ceilf(val);
    }


    inline s32 floorS32(f32 val)
    {
        return static_cast<s32>(floorf(val));
    }

    inline s32 ceilS32(f32 val)
    {
        return static_cast<s32>(ceilf(val));
    }

    inline f32 fmod(f32 x, f32 y)
    {
        return ::fmodf(x, y);
    }

    inline s32 strlen_s32(const Char* str)
    {
        return static_cast<s32>(::strlen(str));
    }

    /**
    @brief Convert IEEE 754 single precision float to half precision float
    */
    u16 toFloat16(f32 f);

    /**
    @brief Convert IEEE 754 half precision float to single precision float
    */
    f32 toFloat32(u16 s);

    s8 floatTo8SNORM(f32 f);
    u8 floatTo8UNORM(f32 f);
    s16 floatTo16SNORM(f32 f);
    u16 floatTo16UNORM(f32 f);

    u8 populationCount(u8 v);
    u32 populationCount(u32 v);

    /**
    */
    u32 mostSignificantBit(u32 val);

    /**
    */
    u32 leastSignificantBit(u32 val);

    u32 bitreverse(u32 x);
    u64 bitreverse(u64 x);

    u32 leadingzero(u32 x);

    f32 calcFOVY(f32 height, f32 znear);

    class auto_buffer
    {
    public:
        explicit auto_buffer(void* x)
            :x_(x)
        {}
        ~auto_buffer()
        {
            GFX_FREE(x_);
        }

        bool valid() const
        {
            return LRENDER_NULL != x_;
        }

        template<class T>
        operator const T*() const
        {
            return x_;
        }

        template<class T>
        operator T*()
        {
            return reinterpret_cast<T*>(x_);
        }

        template<class T>
        const T& operator[](s32 index) const
        {
            return x_[index];
        }

        template<class T>
        T& operator[](s32 index)
        {
            return x_[index];
        }
    private:
        auto_buffer(const auto_buffer&) = delete;
        auto_buffer(auto_buffer&&) = delete;
        auto_buffer& operator=(const auto_buffer&) = delete;
        auto_buffer& operator=(auto_buffer&&) = delete;

        void* x_;
    };

    template<class T>
    class auto_ptr
    {
    public:
        explicit auto_ptr(T* x)
            :x_(x)
        {}
        ~auto_ptr()
        {
            GFX_DELETE(x_);
        }

        operator const T*() const
        {
            return x_;
        }
        operator T*()
        {
            return x_;
        }

        const T* operator->() const
        {
            return x_;
        }
        T* operator->()
        {
            return x_;
        }

        const T& operator&() const
        {
            return *x_;
        }
        T& operator&()
        {
            return *x_;
        }
    private:
        auto_ptr(const auto_ptr&) = delete;
        auto_ptr(auto_ptr&&) = delete;
        auto_ptr& operator=(const auto_ptr&) = delete;
        auto_ptr& operator=(auto_ptr&&) = delete;

        T* x_;
    };

    template<class T>
    class auto_array_ptr
    {
    public:
        explicit auto_array_ptr(T* x)
            :x_(x)
        {}
        ~auto_array_ptr()
        {
            GFX_DELETE_ARRAY(x_);
        }

        operator const T*() const
        {
            return x_;
        }
        operator T*()
        {
            return x_;
        }

        const T& operator[](s32 index) const
        {
            return x_[index];
        }

        T& operator[](s32 index)
        {
            return x_[index];
        }
    private:
        auto_array_ptr(const auto_array_ptr&) = delete;
        auto_array_ptr(auto_array_ptr&&) = delete;
        auto_array_ptr& operator=(const auto_array_ptr&) = delete;
        auto_array_ptr& operator=(auto_array_ptr&&) = delete;

        T* x_;
    };

    //---------------------------------------------------------
    //---
    //--- Geometry
    //---
    //---------------------------------------------------------
    class Vector2;
    class Vector3;
    class Vector4;
    class Plane;
    class Sphere;

    void calcAABBPoints(Vector4* AABB, const Vector4& aabbMin, const Vector4& aabbMax);

    //---------------------------------------------------------------------------------
    /**
    @brief 点から平面上の最近傍点を計算
    @param result ... 出力
    @param point ... 入力点
    @param plane ... 入力平面
    @notice 入力平面は正規化されている
    */
    void closestPointPointPlane(Vector3& result, const Vector3& point, const Plane& plane);

    //---------------------------------------------------------------------------------
    /**
    @brief 点から平面への距離を計算
    @return 距離
    @param point ... 入力点
    @param plane ... 入力平面
    @notice 入力平面は正規化されている
    */
    f32 distancePointPlane(f32 x, f32 y, f32 z, const Plane& plane);
    f32 distancePointPlane(const Vector3& point, const Plane& plane);

    //---------------------------------------------------------------------------------
    /**
    @brief 点から線分への最近傍点を計算
    @return result = t*(l1-l0) + l0となるt
    @param result ... 出力
    @param point ... 
    @param l1 ... 
    @param l2 ... 
    */
    f32 closestPointPointSegment(
        Vector3& result,
        const Vector3& point,
        const Vector3& l0,
        const Vector3& l1);

    //---------------------------------------------------------------------------------
    /**
    @brief 点から線分への距離を計算
    @return 点から線分への距離
    @param point ... 
    @param l1 ... 
    @param l2 ... 
    */
    f32 distancePointSegmentSqr(
        const Vector3& point,
        const Vector3& l0,
        const Vector3& l1);
    inline f32 distancePointSegment(
        const Vector3& point,
        const Vector3& l0,
        const Vector3& l1)
    {
        return lrender::sqrt( distancePointSegmentSqr(point, l0, l1) );
    }

    //---------------------------------------------------------------------------------
    /**
    @brief 点から直線への最近傍点を計算
    @return result = t*(l1-l0) + l0となるt
    @param result ... 出力
    @param point ... 
    @param l1 ... 
    @param l2 ... 
    */
    f32 closestPointPointLine(
        Vector3& result,
        const Vector3& point,
        const Vector3& l0,
        const Vector3& l1);

    //---------------------------------------------------------------------------------
    /**
    @brief 点から直線への距離を計算
    @return 点から直線への距離
    @param point ... 
    @param l1 ... 
    @param l2 ... 
    */
    f32 distancePointLineSqr(
        const Vector3& point,
        const Vector3& l0,
        const Vector3& l1);
    inline f32 distancePointLine(
        const Vector3& point,
        const Vector3& l0,
        const Vector3& l1)
    {
        return sqrt( distancePointLineSqr(point, l0, l1) );
    }

    //---------------------------------------------------------------------------------
    /**
    @brief 線分と線分の最近傍点を計算
    @return 線分同士の距離の平方
    @param s ... 
    @param t ... 
    */
    f32 closestPointSegmentSegmentSqr(
        f32& s,
        f32& t,
        Vector3& c0,
        Vector3& c1,
        const Vector3& p0,
        const Vector3& q0,
        const Vector3& p1,
        const Vector3& q1);

    //---------------------------------------------------------------------------------
    f32 distancePointAABBSqr(
        const f32* point,
        const f32* bmin,
        const f32* bmax);

    f32 distancePointAABBSqr(const Vector4& point, const Vector4& bmin, const Vector4& bmax);
    f32 distancePointAABBSqr(const Vector3& point, const Vector3& bmin, const Vector3& bmax);
    f32 distancePointAABBSqr(const Vector3& point, const Vector4& bmin, const Vector4& bmax);
    f32 distancePointAABBSqr(const Vector4& point, const Vector3& bmin, const Vector3& bmax);

    //---------------------------------------------------------------------------------
    void closestPointPointAABB(
        Vector4& result,
        const Vector4& point,
        const Vector4& bmin,
        const Vector4& bmax);

    void closestPointPointAABB(
        Vector3& result,
        const Vector3& point,
        const Vector3& bmin,
        const Vector3& bmax);

    //---------------------------------------------------------------------------------
    /**
    @return 球と平面が交差するか
    @param sphere ...
    @param plane ... 
    @notice 接する場合も含む
    */
    bool testSpherePlane(f32 &t, const Sphere& sphere, const Plane& plane);

    //---------------------------------------------------------------------------------
    bool testSphereSphere(const Sphere& sphere0, const Sphere& sphere1);

    //---------------------------------------------------------------------------------
    bool testSphereSphere(f32& distance, const Sphere& sphere0, const Sphere& sphere1);

    //---------------------------------------------------------------------------------
    //AABBの交差判定
    bool testAABBAABB(const Vector4& bmin0, const Vector4& bmax0, const Vector4& bmin1, const Vector4& bmax1);
    s32 testAABBAABB(const lm128 bbox0[2][3], const lm128 bbox1[2][3]);

    //---------------------------------------------------------------------------------
    bool testSphereAABB(const Sphere& sphere, const Vector4& bmin, const Vector4& bmax);

    //---------------------------------------------------------------------------------
    bool testSphereAABB(Vector4& close, const Sphere& sphere, const Vector4& bmin, const Vector4& bmax);

    //---------------------------------------------------------------------------------
    bool testSphereCapsule(const Sphere& sphere, const Vector3& p0, const Vector3& q0, f32 r0);

    //---------------------------------------------------------------------------------
    bool testCapsuleCapsule(
        const Vector3& p0,
        const Vector3& q0,
        f32 r0,
        const Vector3& p1,
        const Vector3& q1,
        f32 r1);

    //---------------------------------------------------------------------------------
    /**
    */
    void clipTriangle(
        s32& numTriangles,
        Vector4* triangles,
        const Plane& plane);


    //---------------------------------------------------------------------------------
    bool testPointInPolygon(const Vector2& point, const Vector2* points, s32 n);

    bool testPointInTriangle(
        f32& b0, f32& b1, f32& b2,
        const Vector2& p,
        const Vector2& p0, const Vector2& p1, const Vector2& p2);

    void barycentricCoordinate(
        f32& b0, f32& b1, f32& b2,
        const Vector2& p,
        const Vector2& p0,
        const Vector2& p1,
        const Vector2& p2);

    void orthonormalBasis(Vector3& binormal0, Vector3& binormal1, const Vector3& normal);
    void orthonormalBasis(Vector4& binormal0, Vector4& binormal1, const Vector4& normal);

    //---------------------------------------------------------
    //---
    //--- Low-Discrepancy
    //---
    //---------------------------------------------------------
    f32 halton(s32 index, s32 prime);
    f32 halton_next(f32 prev, s32 prime);

    f32 vanDerCorput(u32 n, u32 base);

    f32 radicalInverseVanDerCorput(u32 bits, u32 scramble);
    f32 radicalInverseSobol(u32 i, u32 scramble);
    f32 radicalInverseLarcherPillichshammer(u32 i, u32 scramble);

    inline void sobol02(f32& v0, f32& v1, u32 i, u32 scramble0, u32 scramble1)
    {
        v0 = radicalInverseVanDerCorput(i, scramble0);
        v1 = radicalInverseSobol(i, scramble1);
    }

    //---------------------------------------------------------
    //---
    //--- Hash Functions
    //---
    //---------------------------------------------------------
    /**
    */
    //u32 hash_Bernstein(const u8* v, u32 count);

    /**
    */
    u32 hash_FNV1(const u8* v, u32 count);

    /**
    */
    u32 hash_FNV1a(const u8* v, u32 count);

    /**
    */
    u64 hash_FNV1_64(const u8* v, u32 count);

    /**
    */
    u64 hash_FNV1a_64(const u8* v, u32 count);

    /**
    */
    //u32 hash_Bernstein(const Char* str);

    /**
    */
    u32 hash_FNV1a(const Char* str);

    /**
    */
    u64 hash_FNV1a_64(const Char* str);

    //------------------------------------------------------
    //---
    //--- Path
    //---
    //------------------------------------------------------
    class String;

    struct Path
    {
        static std::string getCurrentDirectory();
        static std::string getFullPathName(const Char* path);
        static std::string getModuleFileName();
        static std::string getModuleDirectory();

        static const s32 Exists_No = 0;
        static const s32 Exists_File = 1;
        static const s32 Exists_Directory = 2;

        static bool isSpecial(u32 flags);
        static bool isSpecial(u32 flags, const Char* name);
        static bool exists(const Char* path);
        static bool exists(const String& path);
        static s32 existsType(const Char* path);
        static s32 existsType(const String& path);
        static bool isFile(u32 flags);
        static bool isFile(const Char* path);
        static bool isFile(const String& path);
        static bool isDirectory(u32 flags);
        static bool isDirectory(const Char* path);
        static bool isDirectory(const String& path);

        static bool isNormalDirectory(const Char* path);
        static bool isNormalDirectory(u32 flags, const Char* name);
        static bool isSpecialDirectory(const Char* path);
        static bool isSpecialDirectory(u32 flags, const Char* name);

        static void getCurrentDirectory(String& path);
        static bool setCurrentDirectory(const String& path);
        static bool setCurrentDirectory(const Char* path);
        static bool isRoot(const String& path);
        static bool isRoot(const Char* path);
        static bool isRoot(s32 length, const Char* path);
        static bool isAbsolute(const String& path);
        static bool isAbsolute(const Char* path);
        static bool isAbsolute(s32 length, const Char* path);
        static void chompPathSeparator(String& path);
        static void chompPathSeparator(Char* path);
        static void chompPathSeparator(s32 length, Char* path);
        static s32 chompPathSeparator(const String& path);
        static s32 chompPathSeparator(const Char* path);
        static s32 chompPathSeparator(s32 length, const Char* path);

        static s32 extractDirectoryPath(String& dst, s32 length, const Char* path);

        /**
        @brief パスからディレクトリパス抽出. dstがNULLの場合, パスの長さを返す
        @return dstの長さ
        @param dst ... 出力. ヌル文字込みで十分なサイズがあること
        @param length ... パスの長さ
        @param path ... パス
        */
        static s32 extractDirectoryPath(Char* dst, s32 length, const Char* path);

        /**
        @brief パスからディレクトリ名抽出. dstがNULLの場合, ディレクトリ名の長さを返す
        @return dstの長さ
        @param dst ... 出力. ヌル文字込みで十分なサイズがあること
        @param length ... パスの長さ
        @param path ... パス
        */
        static s32 extractDirectoryName(Char* dst, s32 length, const Char* path);

        /**
        @brief パスからファイル名抽出. dstがNULLの場合, ファイル名の長さを返す
        @return dstの長さ
        @param dst ... 出力. ヌル文字込みで十分なサイズがあること
        @param length ... パスの長さ
        @param path ... パス
        */
        static s32 extractFileName(Char* dst, s32 length, const Char* path);

        /**
        @brief パスからファイル名抽出. dstがNULLの場合, ファイル名の長さを返す
        @return dstの長さ
        @param dst ... 出力. ヌル文字込みで十分なサイズがあること
        @param length ... パスの長さ
        @param path ... パス
        */
        static s32 extractFileNameWithoutExt(Char* dst, s32 length, const Char* path);

        /**
        @brief パスから最初のファイル名抽出.
        @param length ... 出力. 抽出したファイル名の長さ
        @param name ... 出力. 抽出したファイル名
        @param pathLength ... パスの長さ
        @param path ... パス
        */
        static const Char* parseFirstNameFromPath(s32& length, Char* name, s32 pathLength, const Char* path);

        /**
        @brief パスから拡張子抽出
        */
        static const Char* getExtension(s32 length, const Char* path);

        static void getFilename(String& filename, s32 length, const Char* path);
        static void getDirectoryname(String& directoryname, s32 length, const Char* path);
    };

    //---------------------------------------------------------
    //---
    //--- Time
    //---
    //---------------------------------------------------------
    void sleep(u32 milliSeconds);

    /// Get performance counter
    ClockType getPerformanceCounter();

    /// Get performance count per second
    ClockType getPerformanceFrequency();

    /// Calculate duration time from performance count
    f64 calcTime64(ClockType prevTime, ClockType currentTime);

    /// Calculate duration time from performance count
    inline f32 calcTime(ClockType prevTime, ClockType currentTime)
    {
        return static_cast<f32>(calcTime64(prevTime, currentTime));
    }

    /// Get time in milli second
    u32 getTimeMilliSec();

    template<bool enable = true>
    struct Timer
    {
        Timer()
            :time_(0)
            ,count_(0)
            ,totalTime_(0.0f)
        {}

        void start()
        {
            time_ = getPerformanceCounter();
        }

        void stop()
        {
            totalTime_ += calcTime64(time_, getPerformanceCounter());
            ++count_;
        }

        f64 getAverage() const
        {
            return (0 == count_)? 0.0 : totalTime_/count_;
        }

        void reset();

        ClockType time_;
        s32 count_;
        f64 totalTime_;
    };

    template<bool enable>
    void Timer<enable>::reset()
    {
        time_ = 0;
        count_ = 0;
        totalTime_ = 0.0f;
    }

    template<>
    struct Timer<false>
    {
        void start(){}
        void stop(){}
        f64 getAverage() const{return 0.0;}
        void reset(){}
    };

    //--------------------------------------------------------
    //---
    //--- SSE, AVX
    //---
    //--------------------------------------------------------
    lm128 set_m128(f32 x, f32 y, f32 z, f32 w);
    lm128 load3(const f32* v);
    void store3(f32* v, const lm128& r);
}


//--------------------------------------------------------
//---
//--- SSE, AVX
//---
//--------------------------------------------------------
//---------------------------------------------------------------------------------------------------
inline lrender::lm128 load_ss(lrender::f32 x)
{
    return _mm_load_ss(&x);
}

/// v0*v1 + v2
inline lrender::lm128 muladd_ss(const lrender::lm128& v0, const lrender::lm128& v1, const lrender::lm128& v2)
{
    return _mm_fmadd_ss(v0, v1, v2);
}

/// v0*v1 - v2
inline lrender::lm128 mulsub_ss(const lrender::lm128& v0, const lrender::lm128& v1, const lrender::lm128& v2)
{
    return _mm_fmadd_ss(v0, v1, v2);
}

/// -(v0*v1) + v2
inline lrender::lm128 nmuladd_ss(const lrender::lm128& v0, const lrender::lm128& v1, const lrender::lm128& v2)
{
    return _mm_fnmadd_ss(v0, v1, v2);
}

/// -(v0*v1) - v2
inline lrender::lm128 nmulsub_ss(const lrender::lm128& v0, const lrender::lm128& v1, const lrender::lm128& v2)
{
    return _mm_fnmadd_ss(v0, v1, v2);
}

//---------------------------------------------------------------------------------------------------
inline lrender::lm128 operator+(const lrender::lm128& v0, const lrender::lm128& v1)
{
    return _mm_add_ps(v0, v1);
}

inline lrender::lm128 operator-(const lrender::lm128& v0, const lrender::lm128& v1)
{
    return _mm_sub_ps(v0, v1);
}

inline lrender::lm128 operator*(const lrender::lm128& v0, const lrender::lm128& v1)
{
    return _mm_mul_ps(v0, v1);
}

inline lrender::lm128 operator*(lrender::f32 x, const lrender::lm128& v)
{
    return _mm_mul_ps(_mm_set1_ps(x), v);
}

inline lrender::lm128 operator*(const lrender::lm128& v, lrender::f32 x)
{
    return _mm_mul_ps(v, _mm_set1_ps(x));
}

inline lrender::lm128 operator/(const lrender::lm128& v0, const lrender::lm128& v1)
{
    return _mm_div_ps(v0, v1);
}

inline lrender::lm128 operator/(const lrender::lm128& v, lrender::f32 x)
{
    return _mm_div_ps(v, _mm_set1_ps(x));
}

inline lrender::lm128 sqrt(const lrender::lm128& v)
{
    return _mm_sqrt_ps(v);
}

inline lrender::lm128 minimum(const lrender::lm128& v0, const lrender::lm128& v1)
{
    return _mm_min_ps(v0, v1);
}

inline lrender::lm128 maximum(const lrender::lm128& v0, const lrender::lm128& v1)
{
    return _mm_max_ps(v0, v1);
}

/// v0*v1 + v2
inline lrender::lm128 muladd(const lrender::lm128& v0, const lrender::lm128& v1, const lrender::lm128& v2)
{
    return _mm_fmadd_ps(v0, v1, v2);
}

/// v0*v1 - v2
inline lrender::lm128 mulsub(const lrender::lm128& v0, const lrender::lm128& v1, const lrender::lm128& v2)
{
    return _mm_fmadd_ps(v0, v1, v2);
}

/// -(v0*v1) + v2
inline lrender::lm128 nmuladd(const lrender::lm128& v0, const lrender::lm128& v1, const lrender::lm128& v2)
{
    return _mm_fnmadd_ps(v0, v1, v2);
}

/// -(v0*v1) - v2
inline lrender::lm128 nmulsub(const lrender::lm128& v0, const lrender::lm128& v1, const lrender::lm128& v2)
{
    return _mm_fnmadd_ps(v0, v1, v2);
}

lrender::lm128 normalize(const lrender::lm128& v);

lrender::lm128 floor(const lrender::lm128& v);

lrender::lm128 ceil(const lrender::lm128& v);

inline lrender::lm128 load(lrender::f32 x, lrender::f32 y, lrender::f32 z)
{
    return _mm_set_ps(0.0f, z, y, x);
}

inline lrender::lm128 load(lrender::f32 x, lrender::f32 y, lrender::f32 z, lrender::f32 w)
{
    return _mm_set_ps(w, z, y, x);
}

inline lrender::lm128 load(const lrender::f32* v)
{
    return _mm_loadu_ps(v);
}

lrender::lm128 normalize(lrender::f32 x, lrender::f32 y, lrender::f32 z);
lrender::lm128 normalizeLengthSqr(lrender::f32 x, lrender::f32 y, lrender::f32 z, lrender::f32 lengthSqr);
lrender::lm128 normalizeChecked(lrender::f32 x, lrender::f32 y, lrender::f32 z);

lrender::lm128 normalize(lrender::f32 x, lrender::f32 y, lrender::f32 z, lrender::f32 w);
lrender::lm128 normalizeLengthSqr(lrender::f32 x, lrender::f32 y, lrender::f32 z, lrender::f32 w, lrender::f32 lengthSqr);
lrender::lm128 normalizeChecked(lrender::f32 x, lrender::f32 y, lrender::f32 z, lrender::f32 w);

lrender::lm128 cross3(const lrender::lm128& v0, const lrender::lm128& v1);

//---------------------------------------------------------------------------------------------------
inline lrender::lm256 operator+(const lrender::lm256& v0, const lrender::lm256& v1)
{
    return _mm256_add_ps(v0, v1);
}

inline lrender::lm256 operator-(const lrender::lm256& v0, const lrender::lm256& v1)
{
    return _mm256_sub_ps(v0, v1);
}

inline lrender::lm256 operator*(const lrender::lm256& v0, const lrender::lm256& v1)
{
    return _mm256_mul_ps(v0, v1);
}

inline lrender::lm256 operator*(lrender::f32 x, const lrender::lm256& v)
{
    return _mm256_mul_ps(_mm256_set1_ps(x), v);
}

inline lrender::lm256 operator*(const lrender::lm256& v, lrender::f32 x)
{
    return _mm256_mul_ps(v, _mm256_set1_ps(x));
}

inline lrender::lm256 operator/(const lrender::lm256& v0, const lrender::lm256& v1)
{
    return _mm256_div_ps(v0, v1);
}

inline lrender::lm256 operator/(const lrender::lm256& v, lrender::f32 x)
{
    return _mm256_div_ps(v, _mm256_set1_ps(x));
}

inline lrender::lm256 sqrt(const lrender::lm256& v)
{
    return _mm256_sqrt_ps(v);
}

inline lrender::lm256 minimum(const lrender::lm256& v0, const lrender::lm256& v1)
{
    return _mm256_min_ps(v0, v1);
}

inline lrender::lm256 maximum(const lrender::lm256& v0, const lrender::lm256& v1)
{
    return _mm256_max_ps(v0, v1);
}

/// v0*v1 + v2
inline lrender::lm256 muladd(const lrender::lm256& v0, const lrender::lm256& v1, const lrender::lm256& v2)
{
    return _mm256_fmadd_ps(v0, v1, v2);
}

/// v0*v1 - v2
inline lrender::lm256 mulsub(const lrender::lm256& v0, const lrender::lm256& v1, const lrender::lm256& v2)
{
    return _mm256_fmadd_ps(v0, v1, v2);
}

/// -(v0*v1) + v2
inline lrender::lm256 nmuladd(const lrender::lm256& v0, const lrender::lm256& v1, const lrender::lm256& v2)
{
    return _mm256_fnmadd_ps(v0, v1, v2);
}

/// -(v0*v1) - v2
inline lrender::lm256 nmulsub(const lrender::lm256& v0, const lrender::lm256& v1, const lrender::lm256& v2)
{
    return _mm256_fnmadd_ps(v0, v1, v2);
}

lrender::lm256 floor(const lrender::lm256& v);

lrender::lm256 ceil(const lrender::lm256& v);

inline lrender::lm256 load256(
    lrender::f32 x0, lrender::f32 x1, lrender::f32 x2, lrender::f32 x3,
    lrender::f32 x4, lrender::f32 x5, lrender::f32 x6, lrender::f32 x7)
{
    return _mm256_set_ps(x0, x1, x2, x3, x4, x5, x6, x7);
}
inline lrender::lm256 load256(const lrender::f32* v)
{
    return _mm256_loadu_ps(v);
}
#endif //INC_LRENDER_H_
