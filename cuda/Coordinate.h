#ifndef INC_LCUDA_COORDINATE_H__
#define INC_LCUDA_COORDINATE_H__
/**
@file Coordinate.h
@author t-sakai
@date 2019/06/24
*/
#include "lcuda.h"

namespace lcuda
{
    class Coordinate
    {
    public:
        LCUDA_HOST LCUDA_DEVICE inline Coordinate();
        LCUDA_HOST LCUDA_DEVICE inline explicit Coordinate(const float3& normal);
        LCUDA_HOST LCUDA_DEVICE inline Coordinate(const float3& normal, const float3& binormal, const float3& tangent);

        LCUDA_HOST LCUDA_DEVICE float3 worldToLocal(const float3& v) const;
        LCUDA_HOST LCUDA_DEVICE float3 localToWorld(const float3& v) const;


        float3 normal_;
        float3 binormal_;
        float3 tangent_;
    };

    LCUDA_HOST LCUDA_DEVICE inline Coordinate::Coordinate()
    {
    }

    LCUDA_HOST LCUDA_DEVICE inline Coordinate::Coordinate(const float3& normal)
        :normal_(normal)
    {
        orthonormalBasis(binormal_, tangent_, normal_);
    }

    LCUDA_HOST LCUDA_DEVICE inline Coordinate::Coordinate(const float3& normal, const float3& binormal, const float3& tangent)
        :normal_(normal)
        ,binormal_(binormal)
        ,tangent_(tangent)
    {
    }

    /**
    @brief ñ@ê¸Ç™zé≤ï˚å¸ÇÃç¿ïWån
    */
    class LocalCoordinate
    {
    public:
        LCUDA_HOST LCUDA_DEVICE inline static f32 cosTheta(const float3& v);
        LCUDA_HOST LCUDA_DEVICE inline static f32 absCosTheta(const float3& v);
        LCUDA_HOST LCUDA_DEVICE inline static f32 cosTheta2(const float3& v);

        LCUDA_HOST LCUDA_DEVICE inline static f32 sinTheta(const float3& v);
        LCUDA_HOST LCUDA_DEVICE inline static f32 absSinTheta(const float3& v);
        LCUDA_HOST LCUDA_DEVICE inline static f32 sinTheta2(const float3& v);

        LCUDA_HOST LCUDA_DEVICE inline static f32 tanTheta(const float3& v);
        LCUDA_HOST LCUDA_DEVICE inline static f32 absTanTheta(const float3& v);
        LCUDA_HOST LCUDA_DEVICE inline static f32 tanTheta2(const float3& v);
        LCUDA_HOST LCUDA_DEVICE inline static f32 invTanTheta(const float3& v);


        LCUDA_HOST LCUDA_DEVICE inline static f32 cosPhi(const float3& v);
        LCUDA_HOST LCUDA_DEVICE inline static f32 cosPhi2(const float3& v);
        LCUDA_HOST LCUDA_DEVICE inline static f32 sinPhi(const float3& v);
        LCUDA_HOST LCUDA_DEVICE inline static f32 sinPhi2(const float3& v);

        LCUDA_HOST LCUDA_DEVICE inline static bool isSameHemisphere(const float3& lhs, const float3& rhs);
    };

    LCUDA_HOST LCUDA_DEVICE inline f32 LocalCoordinate::cosTheta(const float3& v)
    {
        return v.z;
    }

    LCUDA_HOST LCUDA_DEVICE inline f32 LocalCoordinate::absCosTheta(const float3& v)
    {
        return fabsf(v.z);
    }

    LCUDA_HOST LCUDA_DEVICE inline f32 LocalCoordinate::cosTheta2(const float3& v)
    {
        return v.z * v.z;
    }

    LCUDA_HOST LCUDA_DEVICE inline f32 LocalCoordinate::sinTheta(const float3& v)
    {
        f32 sn2 = sinTheta2(v);
        return (sn2<=0.0f)? 0.0f : sqrt(sn2);
    }

    LCUDA_HOST LCUDA_DEVICE inline f32 LocalCoordinate::absSinTheta(const float3& v)
    {
        return sinTheta(v);
    }

    LCUDA_HOST LCUDA_DEVICE inline f32 LocalCoordinate::sinTheta2(const float3& v)
    {
        return 1.0f - cosTheta2(v);
    }

    LCUDA_HOST LCUDA_DEVICE inline f32 LocalCoordinate::tanTheta(const float3& v)
    {
        f32 sn2 = sinTheta2(v);
        if(sn2<=ANGLE_EPSILON){
            return 0.0f;
        }
        return sqrt(sn2)/v.z;
    }

    LCUDA_HOST LCUDA_DEVICE inline f32 LocalCoordinate::absTanTheta(const float3& v)
    {
        f32 sn2 = sinTheta2(v);
        if(sn2<=ANGLE_EPSILON){
            return 0.0f;
        }
        return sqrt(sn2)/fabsf(v.z);
    }

    LCUDA_HOST LCUDA_DEVICE inline f32 LocalCoordinate::tanTheta2(const float3& v)
    {
        f32 cs2 = cosTheta2(v);
        f32 sn2 = 1.0f - cs2;
        if(sn2<=ANGLE_EPSILON){
            return 0.0f;
        }
        return sn2/cs2;
    }

    LCUDA_HOST LCUDA_DEVICE inline f32 LocalCoordinate::invTanTheta(const float3& v)
    {
        return 1.0f/tanTheta(v);
    }

    LCUDA_HOST LCUDA_DEVICE inline f32 LocalCoordinate::cosPhi(const float3& v)
    {
        f32 snTheta = sinTheta(v);
        if(snTheta<=ANGLE_EPSILON){
            return 1.0f;
        }
        return clamp(v.x/snTheta, -1.0f, 1.0f);
    }

    LCUDA_HOST LCUDA_DEVICE inline f32 LocalCoordinate::cosPhi2(const float3& v)
    {
        return clamp01(v.x*v.x/sinTheta2(v));
    }

    LCUDA_HOST LCUDA_DEVICE inline f32 LocalCoordinate::sinPhi(const float3& v)
    {
        f32 snTheta = sinTheta(v);
        if(snTheta<=ANGLE_EPSILON){
            return 1.0f;
        }
        return clamp(v.y/snTheta, -1.0f, 1.0f);
    }

    LCUDA_HOST LCUDA_DEVICE inline f32 LocalCoordinate::sinPhi2(const float3& v)
    {
        return clamp01(v.y*v.y/sinTheta2(v));
    }

    LCUDA_HOST LCUDA_DEVICE inline bool LocalCoordinate::isSameHemisphere(const float3& lhs, const float3& rhs)
    {
        const u32& lu = *(u32*)(&lhs.z);
        const u32& ru = *(u32*)(&rhs.z);
        u32 r = lu ^ ru;
        return 0 == (r & ~0x7FFFFFFFU);
    }
}
#endif //INC_LCUDA_COORDINATE_H__
