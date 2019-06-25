/**
@file Coordinate.cpp
@author t-sakai
@date 2019/06/24
*/
#include "Coordinate.h"

namespace lcuda
{
    LCUDA_HOST LCUDA_DEVICE float3 Coordinate::worldToLocal(const float3& v) const
    {
        return ::make_float3(dot(v, binormal_), dot(v, tangent_), dot(v, normal_));
    }

    LCUDA_HOST LCUDA_DEVICE float3 Coordinate::localToWorld(const float3& v) const
    {
        return ::make_float3(
            binormal_.x*v.x + tangent_.x*v.y + normal_.x*v.z,
            binormal_.y*v.x + tangent_.y*v.y + normal_.y*v.z,
            binormal_.z*v.x + tangent_.z*v.y + normal_.z*v.z);
    }
}
