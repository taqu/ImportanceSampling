#ifndef INC_GFX_SPHERE_H__
#define INC_GFX_SPHERE_H__
/**
@file Sphere.h
@author t-sakai
@date 2019/04/29 create
*/
#include "Vector.h"

namespace lrender
{
    /// 球
    class Sphere
    {
    public:
        Sphere()
        {}

        Sphere(const Vector3& position, f32 radius);
        Sphere(f32 x, f32 y, f32 z, f32 radius);

        void set(f32 x, f32 y, f32 z, f32 radius);

        void zero();

        void setPosition(f32 x, f32 y, f32 z);
        void setPosition(const Vector3& position);
        void setPosition(const Vector4& position);

        void setRadius(f32 radius);

        void translate(const Vector3& position);
        void translate(const Vector4& position);

        /**
        @brief 点に外接する球を計算
        */
        static Sphere circumscribed(const Vector3& p0, const Vector3& p1);

        /**
        @brief 点に外接する球を計算
        */
        static Sphere circumscribed(const Vector3& p0, const Vector3& p1, const Vector3& p2);

        /**
        @brief 点に外接する球を計算
        */
        static Sphere circumscribed(const Vector3& p0, const Vector3& p1, const Vector3& p2, const Vector3& p3);


        static Sphere calcMiniSphere(const Vector3* points, u32 numPoints);

        void combine(const Sphere& s0, const Sphere& s1);
        void add(const Sphere& s1){ combine(*this, s1);}
        void add(const Vector4& s1){ combine(*this, reinterpret_cast<const Sphere&>(s1));}

        f32 signedDistanceSqr(const Vector3& p) const;

        f32 distance(const Sphere& rhs) const;

        void swap(Sphere& rhs);

        const Vector3& position() const
        {
            return *((const Vector3*)&x_);
        }

        Vector3& position()
        {
            return *((Vector3*)&x_);
        }

        const f32& radius() const
        {
            return r_;
        }

        operator const Vector4&() const
        {
            return *((const Vector4*)&x_);
        }
        operator Vector4&()
        {
            return *((Vector4*)&x_);
        }

        void getAABB(lm128& bmin, lm128& bmax) const;

        f32 x_;
        f32 y_;
        f32 z_;
        f32 r_; //Radius
    };

    static_assert(std::is_trivially_copyable<Sphere>::value, "Sphere must be trivially copyable");
}
#endif //INC_GFX_SPHERE_H__
