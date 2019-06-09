#ifndef INC_GFX_PLANE_H__
#define INC_GFX_PLANE_H__
/**
@file Plane.h
@author t-sakai
@date 2019/04/29 create
*/
#include "core.h"
#include "Vector.h"

namespace lrender
{
    /// 平面
    class Plane
    {
    public:
        Plane()
        {}

        Plane(f32 nx, f32 ny, f32 nz, f32 d);
        Plane(const Vector3& point, const Vector3& normal);

        f32 dot(f32 x, f32 y, f32 z) const;

        f32 dot(const Vector3& p) const;

        f32 dot(const Vector4& p) const;

        Vector3 normal() const;

        f32 d() const
        {
            return d_;
        }

        void normalize();

        void translate(f32 x, f32 y, f32 z);
        inline void translate(const Vector3& v){ translate(v.x_, v.y_, v.z_);}
        inline void translate(const Vector4& v){ translate(v.x_, v.y_, v.z_);}

        operator const Vector4&() const
        {
            return *((const Vector4*)&nx_);
        }
        operator Vector4&()
        {
            return *((Vector4*)&nx_);
        }

        f32 nx_; /// 平面の法線
        f32 ny_; /// 平面の法線
        f32 nz_; /// 平面の法線
        f32 d_; /// 原点から平面までの距離
    };

    static_assert(std::is_trivially_copyable<Plane>::value, "Plane must be trivially copyable");
}

#endif //INC_GFX_PLANE_H__
