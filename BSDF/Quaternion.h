#ifndef INC_LRENDER_QUATERNION_H__
#define INC_LRENDER_QUATERNION_H__
/**
@file Quaternion.h
@author t-sakai
@date 2009/09/21 create
*/
#include "core.h"

namespace lrender
{
    class Vector3;
    class Vector4;
    class Matrix34;
    class Matrix44;

    class Quaternion
    {
    public:
        static const Quaternion Identity;

        inline Quaternion();
        Quaternion(f32 w, f32 x, f32 y, f32 z);
        explicit inline Quaternion(const lm128& v);

        void set(f32 w, f32 x, f32 y, f32 z);
        void set(const Vector4& v);

        inline f32 operator[](s32 index) const;
        inline f32& operator[](s32 index);

        void identity();

        void setRotateX(f32 radian);
        void setRotateY(f32 radian);
        void setRotateZ(f32 radian);

        void setRotateXYZ(f32 radx, f32 rady, f32 radz);
        void setRotateXZY(f32 radx, f32 rady, f32 radz);
        void setRotateYZX(f32 radx, f32 rady, f32 radz);
        void setRotateYXZ(f32 radx, f32 rady, f32 radz);
        void setRotateZXY(f32 radx, f32 rady, f32 radz);
        void setRotateZYX(f32 radx, f32 rady, f32 radz);

        void setRotateAxis(const Vector3& axis, f32 radian);
        void setRotateAxis(const Vector4& axis, f32 radian);
        void setRotateAxis(f32 x, f32 y, f32 z, f32 radian);

        void lookAt(const Vector3& eye, const Vector3& at);
        /**
        @param dir ... 正規化済み方向
        */
        void lookAt(const Vector3& dir);
        void lookAt(const Vector4& eye, const Vector4& at);
        /**
        @param dir ... 正規化済み方向
        */
        void lookAt(const Vector4& dir);

        void getDireciton(Vector4& dir) const;

        f32 length() const;
        f32 lengthSqr() const;
        Quaternion operator-() const;

        Quaternion& operator+=(const Quaternion& q);
        Quaternion& operator-=(const Quaternion& q);

        Quaternion& operator*=(f32 f);
        //
        Quaternion& operator*=(const Quaternion& q);

        f32 getRotationAngle() const;

        void getRotationAxis(Vector3& axis) const;
        void getMatrix(Matrix34& mat) const;
        void getMatrix(Matrix44& mat) const;

        void getEulerAngles(f32& x, f32& y, f32& z);

        bool isNan() const;

        void swap(Quaternion& rhs);

        static Quaternion rotateX(f32 radian);
        static Quaternion rotateY(f32 radian);
        static Quaternion rotateZ(f32 radian);

        static Quaternion rotateXYZ(f32 radx, f32 rady, f32 radz);
        static Quaternion rotateXZY(f32 radx, f32 rady, f32 radz);
        static Quaternion rotateYZX(f32 radx, f32 rady, f32 radz);
        static Quaternion rotateYXZ(f32 radx, f32 rady, f32 radz);
        static Quaternion rotateZXY(f32 radx, f32 rady, f32 radz);
        static Quaternion rotateZYX(f32 radx, f32 rady, f32 radz);

        static Quaternion rotateAxis(const Vector3& axis, f32 radian);
        static Quaternion rotateAxis(const Vector4& axis, f32 radian);
        static Quaternion rotateAxis(f32 x, f32 y, f32 z, f32 radian);

        f32 w_, x_, y_, z_;
    };

    //--------------------------------------------
    //---
    //--- Quaternion
    //---
    //--------------------------------------------
    static_assert(std::is_trivially_copyable<Quaternion>::value, "Quaternion should be trivially copyable");

    inline Quaternion::Quaternion()
    {}

    inline Quaternion::Quaternion(const lm128& v)
    {
        _mm_storeu_ps(&w_, v);
    }

    inline f32 Quaternion::operator[](s32 index) const
    {
        LASSERT(0<=index && index < 4);
        return (&w_)[index];
    }

    inline f32& Quaternion::operator[](s32 index)
    {
        LASSERT(0<=index && index < 4);
        return (&w_)[index];
    }

    //--- Quaternion's friend functions
    //--------------------------------------------------
    inline static lm128 load(const Quaternion& q)
    {
        return _mm_loadu_ps(&q.w_);
    }

    inline static void store(Quaternion& q, lm128& r)
    {
        _mm_storeu_ps(&q.w_, r);
    }

    void copy(Quaternion& dst, const Quaternion& src);

    Quaternion invert(const Quaternion& q);

    inline Quaternion conjugate(const Quaternion& q)
    {
        //#if defined(LMATH_USE_SSE)
#if 0
        lm128 mask = _mm_loadu_ps((f32*)QuaternionConjugateMask_);

        lm128 r0 = load(*this);
        r0 = _mm_xor_ps(r0, mask);

        return store(r0);
#else
        return {q.w_, -q.x_, -q.y_, -q.z_};
#endif
    }

    Quaternion normalize(const Quaternion& q);
    Quaternion normalize(const Quaternion& q, f32 squaredLength);

    f32 dot(const Quaternion& q0, const Quaternion& q1);

    Quaternion exp(const Quaternion& q, f32 exponent);

    Quaternion mul(const Quaternion& q0, const Quaternion& q1);

    Quaternion mul(f32 a, const Quaternion& q);
    inline Quaternion mul(const Quaternion& q, f32 a)
    {
        mul(a, q);
    }

    Quaternion mul(const Vector3& v, const Quaternion& q);
    Quaternion mul(const Quaternion& q, const Vector3& v);

    Quaternion mul(const Vector4& v, const Quaternion& q);
    Quaternion mul(const Quaternion& q, const Vector4& v);

    Quaternion rotateToward(const Vector4& from, const Vector4& to);

    /**
    @brief 線形補間。q = (1-t)*q1 + t*q2
    @param q0 ...
    @param q1 ...
    @param t ... 補間比
    */
    Quaternion lerp(const Quaternion& q0, const Quaternion& q1, f32 t);

    /**
    @brief 球面線形補間。q = (1-t)*q1 + t*q2
    @param q0 ...
    @param q1 ...
    @param t ... 補間比
    */
    Quaternion slerp(const Quaternion& q0, const Quaternion& q1, f32 t);

    Quaternion squad(const Quaternion& q0, const Quaternion& q1, const Quaternion& a, const Quaternion& b, f32 t);

}

#endif //INC_LRENDER_QUATERNION_H__
