/**
@file core.cpp
@author t-sakai
@date 2019/02/13
*/
#include "core.h"
#include <Windows.h>
#include "Vector.h"
#include "Plane.h"
#include "Sphere.h"
#include "LString.h"

namespace lrender
{
    GFX_ALIGN32 const f32 One[8] ={1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

    void log(const Char* format, ...)
    {
        LASSERT(LRENDER_NULL != format);

        va_list args;
        va_start(args, format);

#if defined(ANDROID)
        switch(level){
        case Level_Error:
        case Level_Warn:
            __android_log_vprint(ANDROID_LOG_ERROR, "GFX", format, args);
            break;
        case Level_Info:
        default:
            __android_log_vprint(ANDROID_LOG_DEBUG, "GFX", format, args);
            break;
        }
#else

#ifdef _DEBUG
        static const u32 MaxBuffer = 256;
#else
        static const u32 MaxBuffer = 64;
#endif //_DEBUG

        Char buffer[MaxBuffer+2];

#if defined(_WIN32)
        s32 count=vsprintf_s(buffer, MaxBuffer, format, args);
#else
        s32 count = ::vsnprintf(buffer, MaxBuffer, format, args);
#endif //defined(_WIN32)
        va_end(args);
        if(count<=0){
            return;
        }
#if defined(_WIN32)
        buffer[count] = CharLF;
        buffer[count+1] = CharNull;
        OutputDebugString(buffer);
#else
        buffer[count] = CharLF;
        buffer[count+1] = CharNull;
        std::cerr << buffer;
#endif //defined(_WIN32)

#endif //defined(ANDROID)
    }

    template<>
    f32 absolute<f32>(f32 val)
    {
        UnionU32F32 u;
        u.f32_ = val;
        u.u32_ &= 0x7FFFFFFFU;
        return u.f32_;
    }

    f32 sqrt(f32 x)
    {
        f32 ret;
        __m128 tmp = _mm_set_ss(x);
        _mm_store_ss(&ret, _mm_sqrt_ss(tmp));
        return ret;
    }

    f32 clamp01(f32 x)
    {
        UnionS32F32 u;
        u.f32_ = x;

        s32 s = u.s32_ >> 31;
        s = ~s;
        u.s32_ &= s;

        u.f32_ -= 1.0f;
        s = u.s32_ >> 31;
        u.s32_ &= s;
        u.f32_ += 1.0f;
        return u.f32_;
    }

    u16 toFloat16(f32 f)
    {
#if defined(CORE_DISABLE_F16C)
        UnionU32F32 t;
        t.f32_ = f;

        u16 sign = (t.u32_>>16) & 0x8000U;
        s32 exponent = (t.u32_>>23) & 0x00FFU;
        u32 fraction = t.u32_ & 0x007FFFFFU;

        if(exponent == 0){
            return sign; //Signed zero

        }else if(exponent == 0xFFU){
            if(fraction == 0){
                return sign | 0x7C00U; //Signed infinity
            }else{
                return static_cast<u16>((fraction>>13) | 0x7C00U); //NaN
            }
        }else {
            exponent += (-127 + 15);
            if(exponent>=0x1F){ //Overflow
                return sign | 0x7C00U;
            }else if(exponent<=0){ //Underflow
                s32 shift = 14 - exponent;
                if(shift>24){ //Too small
                    return sign;
                }else{
                    fraction |= 0x800000U; //Add hidden bit
                    u16 frac = static_cast<u16>(fraction >> shift);
                    if((fraction>>(shift-1)) & 0x01U){ //Round lowest 1 bit
                        frac += 1;
                    }
                    return sign | frac;
                }
            }
        }

        u16 ret = static_cast<u16>(sign | ((exponent<<10) & 0x7C00U) | (fraction>>13));
        if((fraction>>12) & 0x01U){ //Round lower 1 bit
            ret += 1;
        }
        return ret;
#else
        __declspec(align(16)) u16 result[8];
        _mm_store_si128((__m128i*)result, _mm_cvtps_ph(_mm_set1_ps(f), 0)); //round to nearest
        return result[0];
#endif
    }

    f32 toFloat32(u16 h)
    {
#if defined(CORE_DISABLE_F16C)
        u32 sign = (h & 0x8000U) << 16;
        u32 exponent = ((h & 0x7C00U) >> 10);
        u32 fraction = (h & 0x03FFU);

        if(exponent == 0){
            if(fraction != 0){
                fraction <<= 1;
                while(0==(fraction & 0x0400U)){
                    ++exponent;
                    fraction <<= 1;
                }
                exponent = (127 - 15) - exponent;
                fraction &= 0x03FFU;
            }

        }else if(exponent == 0x1FU){
            exponent = 0xFFU; //Infinity or NaN

        }else{
            exponent += (127 - 15);
        }

        UnionU32F32 t;
        t.u32_ = sign | (exponent<<23) | (fraction<<13);

        return t.f32_;
#else
        __declspec(align(16)) f32 result[4];
        _mm_store_ps(result, _mm_cvtph_ps(_mm_set1_epi16(*(s16*)&h)));
        return result[0];
#endif
    }

    s8 floatTo8SNORM(f32 f)
    {
        s32 s = static_cast<s32>(f*128);
        return static_cast<s8>(clamp(s, -128, 127));
    }

    u8 floatTo8UNORM(f32 f)
    {
        u32 s = static_cast<u32>(f*255 + 0.5f);
        return static_cast<u8>((255<s)? 255 : s);
    }

    s16 floatTo16SNORM(f32 f)
    {
        s32 s = static_cast<s32>(f*32768);
        return static_cast<s16>(clamp(s, -32768, 32767));
    }

    u16 floatTo16UNORM(f32 f)
    {
        u32 s = static_cast<u32>(f*65535 + 0.5f);
        return static_cast<u16>((65535<s)? 65535 : s);
    }

    u8 populationCount(u8 v)
    {
        v = (v & 0x55U) + ((v>>1) & 0x55U);
        v = (v & 0x33U) + ((v>>2) & 0x33U);
        return v = (v & 0x0FU) + ((v>>4) & 0x0FU);
    }

    u32 populationCount(u32 v)
    {
        v = (v & 0x55555555U) + ((v >> 1) & 0x55555555U);
        v = (v & 0x33333333U) + ((v >> 2) & 0x33333333U);
        v = (v & 0x0F0F0F0FU) + ((v >> 4) & 0x0F0F0F0FU);
        v = (v & 0x00FF00FFU) + ((v >> 8) & 0x00FF00FFU);
        return (v & 0x0000FFFFU) + ((v >> 16) & 0x0000FFFFU);
    }

    u32 mostSignificantBit(u32 v)
    {
#if defined(_MSC_VER)
        unsigned long index;
        return (_BitScanReverse(&index, v))? (u32)index : 0;

#elif defined(__GNUC__)
        return (0!=v)? (__builtin_clzl(v) ^ 0x3FU) : 0;
#else
        static const u32 shifttable[] =
        {
            0, 1, 2, 2, 3, 3, 3, 3,
            4, 4, 4, 4, 4, 4, 4, 4,
        };
        u32 ret = 0;

        if(v & 0xFFFF0000U){
            ret += 16;
            v >>= 16;
        }

        if(v & 0xFF00U){
            ret += 8;
            v >>= 8;
        }

        if(v & 0xF0U){
            ret += 4;
            v >>= 4;
        }
        return ret + shifttable[v];
#endif
    }

    u32 leastSignificantBit(u32 v)
    {
#if defined(_MSC_VER)
        unsigned long index;
        return _BitScanForward(&index, v)? (u32)index : 0;

#elif defined(__GNUC__)
        return (0!=v)? (__builtin_ctzl(v)) : 0;
#else
        if(0 == v){
            return 32U;
        }
        u32 count = (v&0xAAAAAAAAU)? 0x01U:0;

        if(v&0xCCCCCCCCU){
            count |= 0x02U;
        }

        if(v&0xF0F0F0F0U){
            count |= 0x04U;
        }

        if(v&0xFF00FF00U){
            count |= 0x08U;
        }

        return 31U-((v&0xFFFF0000U)? count|0x10U:count);
#endif
    }

    //u16 mostSignificantBit(u16 v)
    //{
    //    static const u16 shifttable[] =
    //    {
    //        0, 1, 2, 2, 3, 3, 3, 3,
    //        4, 4, 4, 4, 4, 4, 4, 4,
    //    };
    //    u16 ret = 0;

    //    if(v & 0xFF00U){
    //        ret += 8;
    //        v >>= 8;
    //    }

    //    if(v & 0xF0U){
    //        ret += 4;
    //        v >>= 4;
    //    }
    //    return ret + shifttable[v];
    //}

    //u8 mostSignificantBit(u8 v)
    //{
    //    static const u8 shifttable[] =
    //    {
    //        0, 1, 2, 2, 3, 3, 3, 3,
    //    };
    //    u8 ret = 0;
    //    if(v & 0xF0U){
    //        ret += 4;
    //        v >>= 4;
    //    }

    //    if(v & 0xCU){
    //        ret += 2;
    //        v >>= 2;
    //    }
    //    return ret + shifttable[v];
    //}

    u32 bitreverse(u32 x)
    {
#if defined(__GNUC__)
        x = __builtin_bswap32(x);
#elif defined(__clang__)
        x = __builtin_bswap32(x);
#else
        x = (x << 16) | (x >> 16);
        x = ((x & 0x00FF00FFU) << 8) | ((x & 0xFF00FF00U) >> 8);
#endif
        x = ((x & 0x0F0F0F0FU) << 4) | ((x & 0xF0F0F0F0U) >> 4);
        x = ((x & 0x33333333U) << 2) | ((x & 0xCCCCCCCCU) >> 2);
        x = ((x & 0x55555555U) << 1) | ((x & 0xAAAAAAAAU) >> 1);
        return x;
    }

    u64 bitreverse(u64 x)
    {
#if defined(__GNUC__)
        x = __builtin_bswap64(x);
#elif defined(__clang__)
        x = __builtin_bswap64(x);
#else
        x = (x << 32) | (x >> 32);
        x = ((x & 0x0000FFFF0000FFFFULL) << 16) | ((x & 0xFFFF0000FFFF0000ULL) >> 16);
        x = ((x & 0x00FF00FF00FF00FFULL) << 8) | ((x & 0xFF00FF00FF00FF00ULL) >> 8);
#endif
        x = ((x & 0x0F0F0F0F0F0F0F0FULL) << 4) | ((x & 0xF0F0F0F0F0F0F0F0ULL) >> 4);
        x = ((x & 0x3333333333333333ULL) << 2) | ((x & 0xCCCCCCCCCCCCCCCCULL) >> 2);
        x = ((x & 0x5555555555555555ULL) << 1) | ((x & 0xAAAAAAAAAAAAAAAAULL) >> 1);
        return x;
    }

    // return undefined, if x==0
    u32 leadingzero(u32 x)
    {
#if defined(_MSC_VER)
        DWORD n;
        _BitScanReverse(&n, x);
        return 31-n;
#elif defined(__GNUC__)
        return __builtin_clz(x);
#else
        u32 n = 0;
        if(x<=0x0000FFFFU){ n+=16; x<<=16;}
        if(x<=0x00FFFFFFU){ n+= 8; x<<= 8;}
        if(x<=0x0FFFFFFFU){ n+= 4; x<<= 4;}
        if(x<=0x3FFFFFFFU){ n+= 2; x<<= 2;}
        if(x<=0x7FFFFFFFU){ ++n;}
        return n;
#endif
    }

    f32 calcFOVY(f32 height, f32 znear)
    {
        return 2.0f*atan2(0.5f*height, znear);
    }

    //---------------------------------------------------------
    //---
    //--- Geometry
    //---
    //---------------------------------------------------------
    void calcAABBPoints(Vector4* AABB, const Vector4& aabbMin, const Vector4& aabbMax)
    {
        AABB[0].set(aabbMax.x_, aabbMax.y_, aabbMin.z_, 1.0f);
        AABB[1].set(aabbMin.x_, aabbMax.y_, aabbMin.z_, 1.0f);
        AABB[2].set(aabbMax.x_, aabbMin.y_, aabbMin.z_, 1.0f);
        AABB[3].set(aabbMin.x_, aabbMin.y_, aabbMin.z_, 1.0f);

        AABB[4].set(aabbMax.x_, aabbMax.y_, aabbMax.z_, 1.0f);
        AABB[5].set(aabbMin.x_, aabbMax.y_, aabbMax.z_, 1.0f);
        AABB[6].set(aabbMax.x_, aabbMin.y_, aabbMax.z_, 1.0f);
        AABB[7].set(aabbMin.x_, aabbMin.y_, aabbMax.z_, 1.0f);
    }

    //---------------------------------------------------------------------------------
    // 点から平面上の最近傍点を計算
    void closestPointPointPlane(Vector3& result, const Vector3& point, const Plane& plane)
    {
        f32 t = plane.dot(point);
        result.set(plane.nx_, plane.ny_, plane.nz_);
        result *= -t;
        result += point;
    }

    //---------------------------------------------------------------------------------
    // 点から平面への距離を計算
    f32 distancePointPlane(f32 x, f32 y, f32 z, const Plane& plane)
    {
        return plane.dot(x, y, z);
    }
    f32 distancePointPlane(const Vector3& point, const Plane& plane)
    {
        return plane.dot(point);
    }

    //---------------------------------------------------------------------------------
    // 点から線分への最近傍点を計算
    f32 closestPointPointSegment(
        Vector3& result,
        const Vector3& point,
        const Vector3& l0,
        const Vector3& l1)
    {
        Vector3 v0 = l1-l0;
        Vector3 v1 = point - l0;

        f32 t = dot(v1, v0);
        if(t<=F32_EPSILON){
            t = 0.0f;
            result = l0;
        }else{
            f32 denom = v0.lengthSqr();
            if(denom<=t){
                t = 1.0f;
                result = l1;
            }else{
                t /= denom;
                result = v0;
                result *= t;
                result += l0;
            }
        }
        return t;
    }

    //---------------------------------------------------------------------------------
    // 点から線分への距離を計算
    f32 distancePointSegmentSqr(
        const Vector3& point,
        const Vector3& l0,
        const Vector3& l1)
    {
        Vector3 v0 = l1 - l0;
        Vector3 v1 = point - l0;

        f32 t = dot(v0, v1);
        if(t<=F32_EPSILON){
            return v1.lengthSqr();
        }else{
            f32 u = v0.lengthSqr();
            if(u<=t){
                return distanceSqr(point, l1);
            }else{
                return v1.lengthSqr() - t*t/u;
            }
        }
    }

    //---------------------------------------------------------------------------------
    // 点から直線への最近傍点を計算
    f32 closestPointPointLine(
        Vector3& result,
        const Vector3& point,
        const Vector3& l0,
        const Vector3& l1)
    {
        Vector3 v0 = l1 - l0;
        Vector3 v1 = point - l0;

        f32 t;
        f32 denom = v0.lengthSqr();
        if(denom<=F32_EPSILON){
            t = 0.0f;
            result = l0;
        } else{
            t = dot(v1, v0);
            t /= denom;
            result = v0;
            result *= t;
            result += l0;
        }
        return t;
    }

    //---------------------------------------------------------------------------------
    // 点から直線への距離を計算
    f32 distancePointLineSqr(
        const Vector3& point,
        const Vector3& l0,
        const Vector3& l1)
    {
        Vector3 v0 = l1 - l0;
        Vector3 v1 = point - l0;

        f32 denom = v0.lengthSqr();
        if(denom<=F32_EPSILON){
            return v1.lengthSqr();
        }else{
            f32 t = dot(v1, v0);
            return v1.lengthSqr() - t*t/denom;
        }
    }

    //---------------------------------------------------------------------------------
    // 線分と線分の最近傍点を計算
    f32 closestPointSegmentSegmentSqr(
        f32& s,
        f32& t,
        Vector3& c0,
        Vector3& c1,
        const Vector3& p0,
        const Vector3& q0,
        const Vector3& p1,
        const Vector3& q1)
    {
        Vector3 d0 = q0-p0;
        Vector3 d1 = q1-p1;
        Vector3 r = p0-p1;
        f32 a = dot(d0, d0);
        f32 e = dot(d1, d1);
        f32 f = dot(d1, r);
        if(a<=F32_EPSILON && e<=F32_EPSILON){
            s = t = 0.0f;
            c0 = p0;
            c1 = p1;
            return dot(r, r);
        }
        if(a<=F32_EPSILON){
            s = 0.0f;
            t = clamp01(f/e);
        }else if(e<=F32_EPSILON){
            t = 0.0f;
            s = clamp01(-dot(d0,r)/a);
        }else{
            f32 b = dot(d0,d1);
            f32 c = dot(d0,r);
            f32 denom = a*e - b*b;
            if(F32_EPSILON<denom){
                s = clamp01((b*f-c*e)/denom);
            }else{
                s = 0.0f;
            }
            f32 tnorm = b*s + f;
            if(tnorm<0.0f){
                t = 0.0f;
                s = clamp01(-c/a);
            }else if(e<tnorm){
                t = 1.0f;
                s = clamp01((b-c)/a);
            }else{
                t = tnorm/e;
            }
        }

        c0 = p0 + d0 * s;
        c1 = p1 + d1 * t;
        return distanceSqr(c0, c1);
    }

    //---------------------------------------------------------------------------------
    f32 distancePointAABBSqr(
        const f32* point,
        const f32* bmin,
        const f32* bmax)
    {
        f32 distance = 0.0f;
        for(s32 i=0; i<3; ++i){
            if(point[i]<bmin[i]){
                f32 t = bmin[i] - point[i];
                distance += t*t;

            } else if(bmax[i]<point[i]){
                f32 t = point[i] - bmax[i];
                distance += t*t;
            }
        }
        return distance;
    }

    f32 distancePointAABBSqr(const Vector4& point, const Vector4& bmin, const Vector4& bmax)
    {
        return distancePointAABBSqr(&point.x_, &bmin.x_, &bmax.x_);
    }

    f32 distancePointAABBSqr(const Vector3& point, const Vector3& bmin, const Vector3& bmax)
    {
        return distancePointAABBSqr(&point.x_, &bmin.x_, &bmax.x_);
    }

    f32 distancePointAABBSqr(const Vector3& point, const Vector4& bmin, const Vector4& bmax)
    {
        return distancePointAABBSqr(&point.x_, &bmin.x_, &bmax.x_);
    }

    f32 distancePointAABBSqr(const Vector4& point, const Vector3& bmin, const Vector3& bmax)
    {
        return distancePointAABBSqr(&point.x_, &bmin.x_, &bmax.x_);
    }

    //---------------------------------------------------------------------------------
    void closestPointPointAABB(
        Vector4& result,
        const Vector4& point,
        const Vector4& bmin,
        const Vector4& bmax)
    {
        result.w_ = 0.0f;
        for(s32 i=0; i<3; ++i){
            f32 v = point[i];
            if(v<bmin[i]){
                v = bmin[i];

            }else if(bmax[i]<v){
                v = bmax[i];
            }
            result[i] = v;
        }
    }

    //---------------------------------------------------------------------------------
    void closestPointPointAABB(
        Vector3& result,
        const Vector3& point,
        const Vector3& bmin,
        const Vector3& bmax)
    {
        for(s32 i=0; i<3; ++i){
            f32 v = point[i];
            if(v<bmin[i]){
                v = bmin[i];

            }else if(bmax[i]<v){
                v = bmax[i];
            }
            result[i] = v;
        }
    }

    //---------------------------------------------------------------------------------
    // 球と平面が交差するか
    bool testSpherePlane(f32 &t, const Sphere& sphere, const Plane& plane)
    {
        t = distancePointPlane(sphere.x_, sphere.y_, sphere.z_, plane);
        return (absolute(t)<=sphere.r_);
    }

    //---------------------------------------------------------------------------------
    bool testSphereSphere(const Sphere& sphere0, const Sphere& sphere1)
    {
        f32 distance = sphere0.distance(sphere1);
        f32 radius = sphere0.r_ + sphere1.r_ + F32_EPSILON;

        return (distance <= radius);
    }

    //---------------------------------------------------------------------------------
    bool testSphereSphere(f32& distance, const Sphere& sphere0, const Sphere& sphere1)
    {
        distance = sphere0.distance(sphere1);
        f32 radius = sphere0.r_ + sphere1.r_ + F32_EPSILON;

        return (distance <= radius);
    }

    //---------------------------------------------------------------------------------
    //AABBの交差判定
    bool testAABBAABB(const Vector4& bmin0, const Vector4& bmax0, const Vector4& bmin1, const Vector4& bmax1)
    {
        if(bmax0.x_<bmin1.x_ || bmin0.x_>bmax1.x_){
            return false;
        }

        if(bmax0.y_<bmin1.y_ || bmin0.y_>bmax1.y_){
            return false;
        }

        if(bmax0.z_<bmin1.z_ || bmin0.z_>bmax1.z_){
            return false;
        }
        return true;
    }

    s32 testAABBAABB(const lm128 bbox0[2][3], const lm128 bbox1[2][3])
    {
        u32 mask = 0xFFFFFFFFU;
        f32 fmask = *((f32*)&mask);

        lm128 t = _mm_set1_ps(fmask);
        for(s32 i=0; i<3; ++i){
            t = _mm_and_ps(t, _mm_cmple_ps(bbox0[0][i], bbox1[1][i]));
            t = _mm_and_ps(t, _mm_cmple_ps(bbox1[0][i], bbox0[1][i]));
        }
        return _mm_movemask_ps(t);
    }

    //---------------------------------------------------------------------------------
    bool testSphereAABB(const Sphere& sphere, const Vector4& bmin, const Vector4& bmax)
    {
        f32 distance = distancePointAABBSqr(sphere, bmin, bmax);
        return distance <= (sphere.radius()*sphere.radius());
    }

    //---------------------------------------------------------------------------------
    bool testSphereAABB(Vector4& close, const Sphere& sphere, const Vector4& bmin, const Vector4& bmax)
    {
        closestPointPointAABB(close, sphere, bmin, bmax);

        Vector4 d = close - sphere;
        d.w_ = 0.0f;
        return d.lengthSqr() <= (sphere.radius()*sphere.radius());
    }

    //---------------------------------------------------------------------------------
    bool testSphereCapsule(const Sphere& sphere, const Vector3& p0, const Vector3& q0, f32 r0)
    {
        f32 distanceSqr = distancePointSegmentSqr(sphere.position(), p0, q0);
        f32 radius = sphere.radius() + r0;
        return distanceSqr <= (radius*radius);
    }

    //---------------------------------------------------------------------------------
    bool testCapsuleCapsule(
        const Vector3& p0,
        const Vector3& q0,
        f32 r0,
        const Vector3& p1,
        const Vector3& q1,
        f32 r1)
    {
        f32 s,t;
        Vector3 c0,c1;
        f32 distanceSqr = closestPointSegmentSegmentSqr(
            s,t,
            c0, c1,
            p0, q0,
            p1, q1);
        f32 radius = r0 + r1;
        return distanceSqr <= (radius*radius);
    }

    //---------------------------------------------------------------------------------
    void clipTriangle(
        s32& numTriangles,
        Vector4* triangles,
        const Plane& plane)
    {
        Vector4* triangle = &triangles[3*numTriangles-3];
        f32 d[3];
        s32 indices[3];

        s32 inside = 0;
        s32 outside = 3;
        //Determine whether a point is in at the front of the plane.
        for(s32 i=0; i<3; ++i){
            f32 pd = plane.dot(triangle[i]);
            if(pd<0.0f){
                indices[inside] = i;
                d[inside] = pd;
                ++inside;
            }else{
                --outside;
                indices[outside] = i;
                d[outside] = pd;
            }
        }

        switch(inside)
        {
        case 0:
            --numTriangles;
            break;

        case 1:
            {
                Vector4& p0 = triangle[indices[0]];
                Vector4& p1 = triangle[indices[2]];
                Vector4& p2 = triangle[indices[1]];

                f32 t;
                t = d[0] / (d[0] - d[2]);
                Vector4 new0(lerp(p0, p1, t));

                t = d[0] / (d[0] - d[1]);
                Vector4 new1(lerp(p0, p2, t));

                triangle[0] = p0;
                triangle[1] = new0;
                triangle[2] = new1;
            }
            break;

        case 2:
            {
                if(isEqual(d[2], 0.0f)){
                    return;
                }
                Vector4& p0 = triangle[indices[2]];
                Vector4& p1 = triangle[indices[0]];
                Vector4& p2 = triangle[indices[1]];

                f32 t;
                t = d[2] / (d[2] - d[0]);
                Vector4 new0(lerp(p0, p1, t));

                t = d[2] / (d[2] - d[1]);
                Vector4 new1(lerp(p0, p2, t));

                triangle[0] = p1;
                triangle[1] = new0;
                triangle[2] = p2;

                triangles[3*numTriangles + 0] = p2;
                triangles[3*numTriangles + 1] = new0;
                triangles[3*numTriangles + 2] = new1;
                ++numTriangles;
            }
            break;
        };//switch(inside)
    }

    bool testPointInPolygon(const Vector2& point, const Vector2* points, s32 n)
    {
        LASSERT(2<n);
        s32 i0=n-1,i1=0;
        bool yflag0 = (point.y_ <= points[i0].y_);

        bool flag = false;
        for(; i1<n; i0=i1,++i1){
            bool yflag1 = (point.y_ <= points[i1].y_);
            if(yflag0 != yflag1){
                if(yflag0 != yflag1) {
                    if(((point.x_-points[i0].x_)*(points[i1].y_ - points[i0].y_) <= (points[i1].x_ - points[i0].x_) * (point.y_ - points[i0].y_)) == yflag1) {
                        flag = !flag;
                    }
                }
            }
            yflag0 = yflag1;
        }
        return flag;
    }

    bool testPointInTriangle(
        f32& b0, f32& b1, f32& b2,
        const Vector2& p,
        const Vector2& p0, const Vector2& p1, const Vector2& p2)
    {
        f32 t00 = p0.x_-p2.x_;
        f32 t01 = p1.x_-p2.x_;
        f32 t10 = p0.y_-p2.y_;
        f32 t11 = p1.y_-p2.y_;
        f32 determinant = t00*t11 - t10*t01;
        if(isZero(determinant)){
            b0 = 1.0f;
            b1 = 0.0f;
            b2 = 0.0f;
            return false;
        }

        f32 invDet = 1.0f/determinant;
        b0 = ((p1.y_-p2.y_)*(p.x_-p2.x_) + (p2.x_-p1.x_)*(p.y_-p2.y_))*invDet;
        b1 = ((p2.y_-p0.y_)*(p.x_-p2.x_) + (p0.x_-p2.x_)*(p.y_-p2.y_))*invDet;
        b2 = 1.0f-b0-b1;
        return (0.0f<=b0) && (0.0f<=b1) && (0.0f<=b2);
    }

    void barycentricCoordinate(
        f32& b0, f32& b1, f32& b2,
        const Vector2& p,
        const Vector2& p0,
        const Vector2& p1,
        const Vector2& p2)
    {
        f32 t00 = p0.x_-p2.x_;
        f32 t01 = p1.x_-p2.x_;
        f32 t10 = p0.y_-p2.y_;
        f32 t11 = p1.y_-p2.y_;
        f32 determinant = t00*t11 - t10*t01;
        if(isZero(determinant)){
            b0 = 1.0f;
            b1 = 0.0f;
            b2 = 0.0f;
            return;
        }

        f32 invDet = 1.0f/determinant;
        b0 = ((p1.y_-p2.y_)*(p.x_-p2.x_) + (p2.x_-p1.x_)*(p.y_-p2.y_))*invDet;
        b1 = ((p2.y_-p0.y_)*(p.x_-p2.x_) + (p0.x_-p2.x_)*(p.y_-p2.y_))*invDet;
        b2 = 1.0f-b0-b1;
    }

    void orthonormalBasis(Vector3& binormal0, Vector3& binormal1, const Vector3& normal)
    {
        if(absolute(normal.y_)<absolute(normal.x_)){
            f32 invLen = 1.0f/sqrt(normal.x_*normal.x_ + normal.z_*normal.z_);
            binormal1.set(normal.z_*invLen, 0.0f, -normal.x_*invLen);

        }else{
            f32 invLen = 1.0f/sqrt(normal.y_*normal.y_ + normal.z_*normal.z_);
            binormal1.set(0.0f, normal.z_*invLen, -normal.y_*invLen);
        }
        binormal0 = cross(binormal1, normal);
    }

    void orthonormalBasis(Vector4& binormal0, Vector4& binormal1, const Vector4& normal)
    {
        if(absolute(normal.y_)<absolute(normal.x_)){
            f32 invLen = 1.0f/sqrt(normal.x_*normal.x_ + normal.z_*normal.z_);
            binormal1.set(normal.z_*invLen, 0.0f, -normal.x_*invLen, 0.0f);

        }else{
            f32 invLen = 1.0f/sqrt(normal.z_*normal.z_ + normal.z_*normal.z_);
            binormal1.set(0.0f, normal.z_*invLen, -normal.y_*invLen, 0.0f);
        }
        binormal0 = Vector4(cross3(binormal1, normal));
    }

    //---------------------------------------------------------
    //---
    //--- Low-Discrepancy
    //---
    //---------------------------------------------------------
    f32 halton(s32 index, s32 prime)
    {
        f32 result = 0.0f;
        f32 f = 1.0f / prime;
        int i = index;
        while(0 < i) {
            result = result + f * (i % prime);
            i = (s32)floor((f32)i / prime);
            f = f / prime;
        }
        return result;
    }

    f32 halton_next(f32 prev, s32 prime)
    {
        float r = 1.0f - prev - 0.000001f;
        float f = 1.0f/prime;
        if(f < r) {
            return prev + f;
        } else {
            float h = f;
            float hh;
            do {
                hh = h;
                h *= f;
            } while(h >= r);
            return prev + hh + h - 1.0f;
        }
    }

    f32 vanDerCorput(u32 n, u32 base)
    {
        f32 vdc = 0.0f;
        f32 inv = 1.0f/base;
        f32 factor = inv;

        while(n){
            vdc += static_cast<f32>(n%base) * factor;
            n /= base;
            factor *= inv;
        }
        return vdc;
    }

    f32 radicalInverseVanDerCorput(u32 bits, u32 scramble)
    {
        bits = bitreverse(bits);
        bits ^= scramble;
        return static_cast<f32>(bits) / static_cast<f32>(0x100000000L);
    }

    f32 radicalInverseSobol(u32 i, u32 scramble)
    {
        for(u32 v=1U<<31; i; i>>=1, v ^= v>>1){
            if(i&1){
                scramble ^= v;
            }
        }
        return static_cast<f32>(scramble) / static_cast<f32>(0x100000000L);
    }

    f32 radicalInverseLarcherPillichshammer(u32 i, u32 scramble)
    {
        for(u32 v=1U<<31; i; i>>=1, v |= v>>1){
            if(i&1){
                scramble ^= v;
            }
        }
        return static_cast<f32>(scramble) / static_cast<f32>(0x100000000L);
    }

    //---------------------------------------------------------
    //---
    //--- Hash Functions
    //---
    //---------------------------------------------------------
    /**
    */
    //u32 hash_Bernstein(const u8* v, u32 count)
    //{
    //    u32 hash = 5381U;

    //    for(u32 i=0; i<count; ++i){
    //        //hash = 33*hash + v[i];
    //        hash = ((hash<<5)+hash) + v[i];
    //    }
    //    return hash;
    //}

    /**
    */
    u32 hash_FNV1(const u8* v, u32 count)
    {
        u32 hash = 2166136261U;

        for(u32 i=0; i<count; ++i){
            hash *= 16777619U;
            hash ^= v[i];
        }
        return hash;
    }

    /**
    */
    u32 hash_FNV1a(const u8* v, u32 count)
    {
        u32 hash = 2166136261U;

        for(u32 i=0; i<count; ++i){
            hash ^= v[i];
            hash *= 16777619U;
        }
        return hash;
    }

    /**
    */
    u64 hash_FNV1_64(const u8* v, u32 count)
    {
        u64 hash = 14695981039346656037ULL;

        for(u32 i=0; i<count; ++i){
            hash *= 1099511628211ULL;
            hash ^= v[i];
        }
        return hash;
    }

    /**
    */
    u64 hash_FNV1a_64(const u8* v, u32 count)
    {
        u64 hash = 14695981039346656037ULL;

        for(u32 i=0; i<count; ++i){
            hash ^= v[i];
            hash *= 1099511628211ULL;
        }
        return hash;
    }

    /**
    */
    //u32 hash_Bernstein(const Char* str)
    //{
    //    u32 hash = 5381U;

    //    while(CharNull != *str){
    //        //hash = 33*hash + static_cast<u8>(*str);
    //        hash = ((hash<<5)+hash) + static_cast<u8>(*str);
    //        ++str;
    //    }
    //    return hash;
    //}

    /**
    */
    u32 hash_FNV1a(const Char* str)
    {
        u32 hash = 2166136261U;

        while(CharNull != *str){
            hash ^= static_cast<u8>(*str);
            hash *= 16777619U;
            ++str;
        }
        return hash;
    }

    /**
    */
    u64 hash_FNV1a_64(const Char* str)
    {
        u64 hash = 14695981039346656037ULL;

        while(CharNull != *str){
            hash ^= static_cast<u8>(*str);
            hash *= 1099511628211ULL;
            ++str;
        }
        return hash;
    }

    //--------------------------------------------
    //---
    //--- String
    //---
    //--------------------------------------------
    namespace
    {
        std::string& replaceDelimiter(std::string& str)
        {
            for(char& c : str){
                if(PathDelimiterWin == c){
                    c = PathDelimiter;
                }
            }
            return str;
        }

        std::wstring& replaceDelimiterW(std::wstring& str)
        {
            for(wchar_t& c : str){
                if(L'\\' == c){
                    c = L'/';
                }
            }
            return str;
        }
    }

    //--------------------------------------------
    //---
    //--- Path
    //---
    //--------------------------------------------
    std::string Path::getCurrentDirectory()
    {
        u32 size = GetCurrentDirectory(0, LRENDER_NULL);
        std::string str;
        str.resize(size);
        GetCurrentDirectory(size, &str[0]);
        return lrender::move(replaceDelimiter(str));
    }

    std::string Path::getFullPathName(const Char* path)
    {
        LASSERT(LRENDER_NULL != path);
        u32 size = GetFullPathName(path, 0, LRENDER_NULL, LRENDER_NULL);
        std::string str;
        str.resize(size);
        GetFullPathName(path, size, &str[0], LRENDER_NULL);
        return lrender::move(replaceDelimiter(str));
    }

    std::string Path::getModuleFileName()
    {
        std::string str;
        u32 buffSize = 64;
        for(s32 i=1; i<=4; ++i){
            str.resize(buffSize);
            u32 size = GetModuleFileName(LRENDER_NULL, &str[0], buffSize);
            if(size<=str.size() && GetLastError() == ERROR_SUCCESS){
                break;
            }
            buffSize <<= 1;
        }
        return lrender::move(replaceDelimiter(str));
    }

    std::string Path::getModuleDirectory()
    {
        std::string str = lrender::move(getModuleFileName());
        std::string::size_type pos = str.find_last_of('/');
        if(pos != str.npos) {
            str.resize(pos);
        }
        return lrender::move(str);
    }

    bool Path::isSpecial(u32 flags)
    {
        static const DWORD checks[] =
        {
            FILE_ATTRIBUTE_HIDDEN,
            FILE_ATTRIBUTE_SYSTEM,
            FILE_ATTRIBUTE_ENCRYPTED,
            FILE_ATTRIBUTE_TEMPORARY,
            FILE_ATTRIBUTE_SPARSE_FILE,
        };
        static const s32 Num = sizeof(checks)/sizeof(DWORD);
        for (int i = 0; i<Num; ++i){
            DWORD check = checks[i] & flags;
            if (check != 0){
                return true;
            }
        }
        return false;
    }

    bool Path::isSpecial(u32 flags, const Char* name)
    {
        LASSERT(LRENDER_NULL != name);
        return isSpecial(flags) || '.' == name[0];
    }

    bool Path::exists(const Char* path)
    {
        LASSERT(LRENDER_NULL != path);
        DWORD ret = GetFileAttributes(path);
        return ((DWORD)-1) != ret ? !isSpecial(ret) : false;
    }

    bool Path::exists(const String& path)
    {
        return exists(path.c_str());
    }

    s32 Path::existsType(const Char* path)
    {
        LASSERT(LRENDER_NULL != path);
        DWORD ret = GetFileAttributes(path);
        if (((DWORD)-1)!=ret){
            if (isSpecial(ret)){
                return Exists_No;
            }
            return FILE_ATTRIBUTE_DIRECTORY==(ret&FILE_ATTRIBUTE_DIRECTORY) ? Exists_Directory : Exists_File;
        }
        return Exists_No;
    }

    s32 Path::existsType(const String& path)
    {
        return existsType(path.c_str());
    }

    bool Path::isFile(u32 flags)
    {
        LASSERT(((DWORD)-1) != flags);
        return isSpecial(flags) ? false : 0 == (flags&FILE_ATTRIBUTE_DIRECTORY);
    }

    bool Path::isFile(const Char* path)
    {
        LASSERT(LRENDER_NULL != path);
        DWORD ret = GetFileAttributes(path);
        if (((DWORD)-1) == ret){
            return false;
        }
        return isFile(ret);
    }

    bool Path::isFile(const String& path)
    {
        return isFile(path.c_str());
    }

    bool Path::isDirectory(u32 flags)
    {
        LASSERT(((DWORD)-1) != flags);
        return isSpecial(flags) ? false : 0 != (flags&FILE_ATTRIBUTE_DIRECTORY);
    }

    bool Path::isDirectory(const Char* path)
    {
        LASSERT(LRENDER_NULL != path);
        DWORD ret = GetFileAttributes(path);
        if (((DWORD)-1) == ret){
            return false;
        }
        return isDirectory(ret);
    }

    bool Path::isDirectory(const String& path)
    {
        return isDirectory(path.c_str());
    }

    bool Path::isNormalDirectory(const Char* path)
    {
        LASSERT(LRENDER_NULL != path);
        DWORD ret = GetFileAttributes(path);
        if (((DWORD)-1) == ret){
            return false;
        }
        s32 pathLen = lrender::strlen_s32(path);
        s32 nameLen = extractDirectoryName(LRENDER_NULL, pathLen, path);
        if (nameLen<=0){
            return false;
        }
        const Char* name = path + pathLen - nameLen;
        return isNormalDirectory(ret, name);
    }

    bool Path::isNormalDirectory(u32 flags, const Char* name)
    {
        LASSERT(LRENDER_NULL != name);
        return Path::isDirectory(flags) && '.' != name[0];
    }

    bool Path::isSpecialDirectory(const Char* path)
    {
        LASSERT(LRENDER_NULL != path);
        DWORD ret = GetFileAttributes(path);
        if (((DWORD)-1) == ret){
            return false;
        }
        s32 pathLen = lrender::strlen_s32(path);
        s32 nameLen = extractDirectoryName(LRENDER_NULL, pathLen, path);
        if (nameLen<=0){
            return false;
        }
        const Char* name = path + pathLen - nameLen;
        return isSpecialDirectory(ret, name);
    }

    bool Path::isSpecialDirectory(u32 flags, const Char* name)
    {
        LASSERT(LRENDER_NULL != name);
        return Path::isDirectory(flags) && '.' == name[0];
    }

    void Path::getCurrentDirectory(String& path)
    {
        s32 size = GetCurrentDirectory(0, LRENDER_NULL);
        if (size<=0){
            path.clear();
            return;
        }
        path.fill(size-1, CharNull);
        GetCurrentDirectory(size, &path[0]);

#if defined(_WIN32)
        path.replace('\\', PathDelimiter);
#endif
    }

    bool Path::setCurrentDirectory(const String& path)
    {
        return TRUE == SetCurrentDirectory(path.c_str());
    }

    bool Path::setCurrentDirectory(const Char* path)
    {
        LASSERT(LRENDER_NULL != path);
        return TRUE == SetCurrentDirectory(path);
    }

    bool Path::isRoot(const String& path)
    {
#if defined(_WIN32)
        if (path.length() != 3){
            return false;
        }
        return isalpha(path[0]) && path[1]==':' && path[2]==PathDelimiter;

#else
        if (path.length() != 1){
            return false;
        }
        return path[0] == LVFS_PATH_SEPARATOR;
#endif
    }

    bool Path::isRoot(const Char* path)
    {
        LASSERT(LRENDER_NULL != path);
        return isRoot(static_cast<s32>(strlen(path)), path);
    }

    bool Path::isRoot(s32 length, const Char* path)
    {
        LASSERT(LRENDER_NULL != path);
#if defined(_WIN32)
        if (3 == length){
            return false;
        }
        return (isalpha(path[0]) && path[1]==':' && path[2]==PathDelimiter);
#else
        if (1 == length){
            return false;
        }
        return path[0] == PathDelimiter;
#endif
    }

    bool Path::isAbsolute(const String& path)
    {
#if defined(_WIN32)
        if (path.length() < 3){
            return false;
        }
        return isalpha(path[0]) && path[1]==':' && path[2]==PathDelimiter;

#else
        if (path.length() < 1){
            return false;
        }
        return path[0] == PathDelimiter;
#endif
    }

    bool Path::isAbsolute(const Char* path)
    {
        LASSERT(LRENDER_NULL != path);
        return isAbsolute(static_cast<s32>(::strlen(path)), path);
    }

    bool Path::isAbsolute(s32 length, const Char* path)
    {
        LASSERT(LRENDER_NULL != path);
#if defined(_WIN32)
        if (length < 3){
            return false;
        }
        return isalpha(path[0]) && path[1] == ':' && path[2] == PathDelimiter;

#else
        if (length < 1){
            return false;
        }
        return path[0] == PathDelimiter;
#endif
    }

    void Path::chompPathSeparator(String& path)
    {
        if (path.length()<=0 || Path::isRoot(path)){
            return;
        }
        if (path[path.length()-1] == PathDelimiter){
            path.pop_back();
        }
    }

    void Path::chompPathSeparator(Char* path)
    {
        LASSERT(LRENDER_NULL != path);
        Path::chompPathSeparator(static_cast<s32>(strlen(path)), path);
    }

    void Path::chompPathSeparator(s32 length, Char* path)
    {
        LASSERT(LRENDER_NULL != path);
        if (length<=0 || Path::isRoot(length, path)){
            return;
        }
        if (path[length-1] == PathDelimiter){
            path[length-1] = CharNull;
        }
    }

    s32 Path::chompPathSeparator(const String& path)
    {
        if (path.length()<=0 || Path::isRoot(path)){
            return path.length();
        }
        if (path[path.length()-1] == PathDelimiter){
            return path.length()-1;
        }
        return path.length();
    }

    s32 Path::chompPathSeparator(const Char* path)
    {
        LASSERT(LRENDER_NULL != path);
        return chompPathSeparator(static_cast<s32>(strlen(path)), path);
    }

    s32 Path::chompPathSeparator(s32 length, const Char* path)
    {
        LASSERT(LRENDER_NULL != path);
        if (length<=0 || Path::isRoot(length, path)){
            return length;
        }
        if (path[length-1] == PathDelimiter){
            return length-1;
        }
        return length;
    }

    s32 Path::extractDirectoryName(Char* dst, s32 length, const Char* path)
    {
        LASSERT(LRENDER_NULL != path);
        LASSERT(0<=length);
        if (LRENDER_NULL != dst){
            dst[0] = CharNull;
        }
        if (length<=0){
            return 0;
        }

        if (PathDelimiter == path[length-1]){
            --length;
            if (length<=0){
                return 0;
            }
        }
        s32 i = length-1;
        for (; 0<=i; --i){
            if (PathDelimiter == path[i]){
                break;
            }
        }

        s32 dstLen = length-i-1;
        if (LRENDER_NULL != dst){
            for (s32 j = i+1; j<length; ++j){
                dst[j-i-1] = path[j];
            }
            dst[dstLen] = CharNull;
        }
        return dstLen;
    }

    s32 Path::extractDirectoryPath(String& dst, s32 length, const Char* path)
    {
        LASSERT(LRENDER_NULL != path);
        LASSERT(0<=length);
        if(length<=0){
            dst.clear();
            return 0;
        }

        s32 i = length-1;
        for(; 0<=i; --i){
            if(PathDelimiter == path[i]){
                break;
            }
        }
        dst.reserve(i);
        s32 dstLen = i + 1;
        for(s32 j = 0; j < dstLen; ++j) {
            dst.append(path[j]);
        }
        return dstLen;
    }

    // Extract directory path from path
    s32 Path::extractDirectoryPath(Char* dst, s32 length, const Char* path)
    {
        LASSERT(LRENDER_NULL != path);
        LASSERT(0<=length);
        if (length<=0){
            if (LRENDER_NULL != dst){
                dst[0] = CharNull;
            }
            return 0;
        }

        s32 i = length-1;
        for (; 0<=i; --i){
            if (PathDelimiter == path[i]){
                break;
            }
        }
        s32 dstLen = i+1;
        if (LRENDER_NULL != dst){
            for (s32 j = 0; j<dstLen; ++j){
                dst[j] = path[j];
            }
            dst[dstLen] = CharNull;
        }
        return dstLen;
    }

    //-------------------------------------------------------------
    // パスからファイル名抽出
    s32 Path::extractFileName(Char* dst, s32 length, const Char* path)
    {
        LASSERT(LRENDER_NULL != path);
        LASSERT(0<=length);
        if (length<=0){
            if (LRENDER_NULL != dst){
                dst[0] = CharNull;
            }
            return 0;
        }

        s32 i = length-1;
        for (; 0<=i; --i){
            if (PathDelimiter == path[i]){
                break;
            }
        }

        s32 dstLen = length-i-1;
        if (LRENDER_NULL != dst){
            for (s32 j = i+1; j<length; ++j){
                dst[j-i-1] = path[j];
            }
            dst[dstLen] = CharNull;
        }
        return dstLen;
    }

    // パスからファイル名抽出
    s32 Path::extractFileNameWithoutExt(Char* dst, s32 length, const Char* path)
    {
        LASSERT(LRENDER_NULL != path);
        LASSERT(0<=length);
        if (length<=0){
            if (LRENDER_NULL != dst){
                dst[0] = CharNull;
            }
            return 0;
        }

        s32 i = length-1;
        for (; 0<=i; --i){
            if (PathDelimiter == path[i]){
                break;
            }
        }

        s32 dstLen = length-i-1;
        if (LRENDER_NULL != dst){
            for (s32 j = length-1; i<j; --j){
                if ('.' == path[j]){
                    dstLen = j-i-1;
                }
                dst[j-i-1] = path[j];
            }
            dst[dstLen] = CharNull;
        }
        return dstLen;
    }

    // パスから最初のファイル名抽出
    const Char* Path::parseFirstNameFromPath(s32& length, Char* name, s32 pathLength, const Char* path)
    {
        LASSERT(LRENDER_NULL != name);
        LASSERT(LRENDER_NULL != path);

        length = 0;
        while (CharNull != *path && length<(pathLength-1)){
            if (PathDelimiter == *path){
                ++path;
                break;
            }
            name[length++] = *path;
            ++path;
        }
        name[length] = CharNull;
        return path;
    }

    // パスから拡張子抽出
    const Char* Path::getExtension(s32 length, const Char* path)
    {
        LASSERT(0<=length);
        LASSERT(LRENDER_NULL != path);
        for (s32 i = length-1; 0<=i; --i){
            if (path[i] == '.'){
                return &path[i+1];
            }
        }
        return &path[length];
    }

    void Path::getFilename(String& filename, s32 length, const Char* path)
    {
        LASSERT(0<=length);
        s32 filenameLength = extractFileName(LRENDER_NULL, static_cast<u32>(length), path);
        if (filenameLength<=0){
            filename.clear();
            return;
        }
        filename.fill(filenameLength);
        extractFileName(&filename[0], length, path);
    }

    void Path::getDirectoryname(String& directoryname, s32 length, const Char* path)
    {
        LASSERT(0<=length);
        s32 directorynameLength = extractDirectoryName(LRENDER_NULL, static_cast<u32>(length), path);
        if (directorynameLength<=0){
            directoryname.clear();
            return;
        }
        directoryname.fill(directorynameLength);
        extractDirectoryName(&directoryname[0], length, path);
    }

    //---------------------------------------------------------
    //---
    //--- Time
    //---
    //---------------------------------------------------------
    void sleep(u32 milliSeconds)
    {
#if defined(_WIN32)
        ::Sleep(milliSeconds);
#else
        timespec ts;
        ts.tv_sec = 0;
        while(1000<milliSeconds){
            ts.tv_sec += 1;
            milliSeconds -= 1000;
        }
        ts.tv_nsec = 1000000L * milliSeconds;
        nanosleep(&ts, NULL);
#endif
    }

    ClockType getPerformanceCounter()
    {
#if defined(_WIN32)
        LARGE_INTEGER count;
        QueryPerformanceCounter(&count);
        return count.QuadPart;
#else
        clock_t t = 0;
        t = clock();
        return t;
#endif
    }

    ClockType getPerformanceFrequency()
    {
#if defined(_WIN32)
        LARGE_INTEGER freq;
        QueryPerformanceFrequency(&freq);
        return freq.QuadPart;
#else
        return CLOCKS_PER_SEC;
#endif
    }

    f64 calcTime64(ClockType prevTime, ClockType currentTime)
    {
        ClockType d = (currentTime>=prevTime)? currentTime - prevTime : (std::numeric_limits<ClockType>::max)() - prevTime + currentTime;
        f64 delta = static_cast<f64>(d)/getPerformanceFrequency();
        return delta;
    }

    u32 getTimeMilliSec()
    {
#if defined(_WIN32)
        DWORD time = timeGetTime();
        return static_cast<u32>(time);
#else
        struct timeval tv;
        gettimeofday(&tv, NULL);

        return static_cast<u32>(tv.tv_sec*1000 + tv.tv_usec/1000);
#endif
    }

    //--------------------------------------------------------
    //---
    //--- SSE, AVX
    //---
    //--------------------------------------------------------
    lm128 set_m128(f32 x, f32 y, f32 z, f32 w)
    {
        lm128 t0 = _mm_load_ss(&x);
        lm128 t1 = _mm_load_ss(&z);

        lm128 ret = _mm_unpacklo_ps(t0, t1);
        t0 = _mm_load_ss(&y);
        t1 = _mm_load_ss(&w);
        return _mm_unpacklo_ps(ret, _mm_unpacklo_ps(t0, t1));
    }

    lm128 load3(const f32* v)
    {
        lm128 t = _mm_load_ss(&v[2]);
        t = _mm_movelh_ps(t, t);
        t = _mm_loadl_pi(t, reinterpret_cast<const __m64*>(v));
        return t;
    }

    void store3(f32* v, const lm128& r)
    {
        _mm_storel_pi(reinterpret_cast<__m64*>(v), r);

        static const u32 Shuffle = 170;
        lm128 t = _mm_shuffle_ps(r, r, Shuffle);
        _mm_store_ss(&v[2], t);
    }
}

lrender::lm128 normalize(const lrender::lm128& r0)
{
    lrender::lm128 r1 = r0;
    lrender::lm128 tmp = _mm_mul_ps(r0, r0);
    tmp = _mm_add_ps(_mm_shuffle_ps(tmp, tmp, 0x4E), tmp);
    tmp = _mm_add_ps(_mm_shuffle_ps(tmp, tmp, 0xB1), tmp);

    tmp = _mm_sqrt_ss(tmp);
    tmp = _mm_shuffle_ps(tmp, tmp, 0);

    r1 = _mm_div_ps(r1, tmp);
    return r1;
}

lrender::lm128 normalize(lrender::f32 x, lrender::f32 y, lrender::f32 z)
{
    lrender::lm128 r0 = _mm_set_ps(0.0f, z, y, x);
    lrender::lm128 r1 = r0;
    r0 = _mm_mul_ps(r0, r0);
    r0 = _mm_add_ps(_mm_shuffle_ps(r0, r0, 0x4E), r0);
    r0 = _mm_add_ps(_mm_shuffle_ps(r0, r0, 0xB1), r0);

    r0 = _mm_sqrt_ss(r0);
    r0 = _mm_shuffle_ps(r0, r0, 0);

    return _mm_div_ps(r1, r0);
}

lrender::lm128 normalizeLengthSqr(lrender::f32 x, lrender::f32 y, lrender::f32 z, lrender::f32 lengthSqr)
{
    lrender::lm128 r0 = _mm_set1_ps(lengthSqr);
    lrender::lm128 r1 = _mm_set_ps(0.0f, z, y, x);
    r0 = _mm_sqrt_ps(r0);
    return _mm_div_ps(r1, r0);
}

lrender::lm128 normalizeChecked(lrender::f32 x, lrender::f32 y, lrender::f32 z)
{
    lrender::f32 l = x*x + y*y + z*z;
    if(lrender::isZeroPositive(l)){
        return _mm_setzero_ps();
    } else{
        return normalizeLengthSqr(x, y, z, l);
    }
}

lrender::lm128 normalize(lrender::f32 x, lrender::f32 y, lrender::f32 z, lrender::f32 w)
{
    lrender::lm128 r0 = _mm_set_ps(w, z, y, x);
    lrender::lm128 r1 = r0;
    r0 = _mm_mul_ps(r0, r0);
    r0 = _mm_add_ps(_mm_shuffle_ps(r0, r0, 0x4E), r0);
    r0 = _mm_add_ps(_mm_shuffle_ps(r0, r0, 0xB1), r0);

    r0 = _mm_sqrt_ss(r0);
    r0 = _mm_shuffle_ps(r0, r0, 0);

    return _mm_div_ps(r1, r0);
}

lrender::lm128 normalizeLengthSqr(lrender::f32 x, lrender::f32 y, lrender::f32 z, lrender::f32 w, lrender::f32 lengthSqr)
{
    lrender::lm128 r0 = _mm_set1_ps(lengthSqr);
    lrender::lm128 r1 = _mm_set_ps(w, z, y, x);
    r0 = _mm_sqrt_ps(r0);
    return _mm_div_ps(r1, r0);
}

lrender::lm128 normalizeChecked(lrender::f32 x, lrender::f32 y, lrender::f32 z, lrender::f32 w)
{
    lrender::f32 l = x*x + y*y + z*z + w*w;
    if(lrender::isZeroPositive(l)){
        return _mm_setzero_ps();
    } else{
        return normalizeLengthSqr(x, y, z, w, l);
    }
}

lrender::lm128 floor(const lrender::lm128& tv0)
{
    lrender::lm128 tv1 = _mm_cvtepi32_ps(_mm_cvttps_epi32(tv0));

    return _mm_sub_ps(tv1, _mm_and_ps(_mm_cmplt_ps(tv0, tv1), _mm_set1_ps(1.0f)));
}

lrender::lm128 ceil(const lrender::lm128& tv0)
{
    lrender::lm128 tv1 = _mm_cvtepi32_ps(_mm_cvttps_epi32(tv0));
    return _mm_add_ps(tv1, _mm_and_ps(_mm_cmplt_ps(tv0, tv1), _mm_set1_ps(1.0f)));
}

lrender::lm128 cross3(const lrender::lm128& v0, const lrender::lm128& v1)
{
    lrender::lm128 xv0 = _mm_shuffle_ps(v0, v0, 0xC9);
    lrender::lm128 xv1 = _mm_shuffle_ps(v1, v1, 0xC9);
    lrender::lm128 tmp0 = _mm_fmsub_ps(v0, xv1, _mm_mul_ps(xv0, v1));

    tmp0 = _mm_shuffle_ps(tmp0, tmp0, 0xC9);
    return tmp0;
}

//---------------------------------------------------------------------------------------------------
lrender::lm256 floor(const lrender::lm256& tv0)
{
    lrender::lm256 tv1 = _mm256_cvtepi32_ps(_mm256_cvttps_epi32(tv0));
    return _mm256_sub_ps(tv1, _mm256_and_ps(_mm256_cmp_ps(tv0, tv1, _CMP_LT_OS), _mm256_set1_ps(1.0f)));
}

lrender::lm256 ceil(const lrender::lm256& tv0)
{
    lrender::lm256 tv1 = _mm256_cvtepi32_ps(_mm256_cvttps_epi32(tv0));
    return _mm256_add_ps(tv1, _mm256_and_ps(_mm256_cmp_ps(tv0, tv1, _CMP_LT_OS), _mm256_set1_ps(1.0f)));
}
