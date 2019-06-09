/**
@file Plane.cpp
@author t-sakai
@date 2019/04/29 create
*/
#include "Plane.h"

#ifndef LMATH_USE_SSE
#define LMATH_USE_SSE
#endif

namespace lrender
{
    Plane::Plane(f32 nx, f32 ny, f32 nz, f32 d)
        :nx_(nx)
        ,ny_(ny)
        ,nz_(nz)
        ,d_(d)
    {}

    Plane::Plane(const Vector3& point, const Vector3& normal)
        :nx_(normal.x_)
        ,ny_(normal.y_)
        ,nz_(normal.z_)
        ,d_(-lrender::dot(point, normal))
    {}

    f32 Plane::dot(f32 x, f32 y, f32 z) const
    {
        return (nx_ * x + ny_ * y + nz_ * z + d_);
    }

    f32 Plane::dot(const Vector3& p) const
    {
        return (nx_ * p.x_ + ny_ * p.y_ + nz_ * p.z_ + d_);
    }

    f32 Plane::dot(const Vector4& p) const
    {
        return lrender::dot((const Vector4&)*this, p);
    }

    Vector3 Plane::normal() const
    {
        return {nx_, ny_, nz_};
    }

    void Plane::normalize()
    {
#if defined(LMATH_USE_SSE)
        lm128 r0 = _mm_loadu_ps(&nx_);
        lm128 r1 = r0;
        r0 = _mm_mul_ps(r0, r0);
        lm128 r2 = _mm_add_ps( _mm_shuffle_ps(r0, r0, 0xE1), r0);
        r2 = _mm_add_ps( _mm_shuffle_ps(r0, r0, 0xE2), r2);

        r2 = _mm_sqrt_ss(r2);
        r2 = _mm_shuffle_ps(r2, r2, 0);
        
        r1 = _mm_div_ps(r1, r2);
        _mm_storeu_ps(&nx_, r1);
#else
        f32 l = nx_*nx_ + ny_*ny_ + nz_*nz_;

        LASSERT( !(lcore::isEqual(l, 0.0f)) );
        l = 1.0f/ lcore::sqrt(l);
        v_ *= l;
#endif
    }

    void Plane::translate(f32 x, f32 y, f32 z)
    {
        d_ -= (x*nx_ + y*ny_ + z*nz_);
    }
}
