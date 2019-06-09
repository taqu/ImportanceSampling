/**
@file Vector.cpp
@author t-sakai
@date 2019/03/03 create
*/
#include "Vector.h"
#include "Quaternion.h"
#include "Matrix.h"

#ifndef LMATH_USE_SSE
#define LMATH_USE_SSE
#endif

namespace lrender
{
    //--------------------------------------------
    //---
    //--- Vector2
    //---
    //--------------------------------------------
    const Vector2 Vector2::Zero(0.0f);
    const Vector2 Vector2::One(1.0f);

    f32 Vector2::lengthSqr() const
    {
        return (x_ * x_ + y_ * y_);
    }

    Vector2& Vector2::operator+=(const Vector2& v)
    {
        x_ += v.x_;
        y_ += v.y_;
        return *this;
    }

    Vector2& Vector2::operator-=(const Vector2& v)
    {
        x_ -= v.x_;
        y_ -= v.y_;
        return *this;
    }

    Vector2& Vector2::operator*=(f32 f)
    {
        x_ *= f;
        y_ *= f;
        return *this;
    }

    Vector2& Vector2::operator/=(f32 f)
    {
        LASSERT(f != 0.0f);
        f = 1.0f/f;
        x_ *= f;
        y_ *= f;
        return *this;
    }

    bool Vector2::isEqual(const Vector2& v) const
    {
        return (lrender::isEqual(x_, v.x_)
            && lrender::isEqual(y_, v.y_));
    }

    bool Vector2::isEqual(const Vector2& v, f32 epsilon) const
    {
        return (lrender::isEqual(x_, v.x_, epsilon)
            && lrender::isEqual(y_, v.y_, epsilon));
    }

    bool Vector2::isNan() const
    {
        return (lrender::isNan(x_) || lrender::isNan(y_));
    }

    Vector2 normalize(const Vector2& v)
    {
        f32 l = v.lengthSqr();
        LASSERT(!lrender::isZero(l));
        l = 1.0f/::sqrtf(l);
        return {v.x_*l, v.y_*l};
    }

    Vector2 normalizeChecked(const Vector2& v)
    {
        f32 l = v.lengthSqr();
        if(lrender::isZero(l)){
            return Vector2::Zero;
        }
        l = 1.0f/::sqrtf(l);
        return {v.x_*l, v.y_*l};
    }

    Vector2 operator+(const Vector2& v0, const Vector2& v1)
    {
        return {v0.x_+v1.x_, v0.y_+v1.y_};
    }

    Vector2 operator-(const Vector2& v0, const Vector2& v1)
    {
        return {v0.x_-v1.x_, v0.y_-v1.y_};
    }

    Vector2 operator*(f32 f, const Vector2& v)
    {
        return {f*v.x_, f*v.y_};
    }

    Vector2 operator*(const Vector2& v, f32 f)
    {
        return {v.x_*f, v.y_*f};
    }

    Vector2 operator/(const Vector2& v, f32 f)
    {
        f32 inv = 1.0f/f;
        return {v.x_*inv, v.y_*inv};
    }

    f32 dot(const Vector2& v0, const Vector2& v1)
    {
        return (v0.x_*v1.x_ + v0.y_*v1.y_);
    }

    f32 distanceSqr(const Vector2& v0, const Vector2& v1)
    {
        f32 dx = v0.x_ - v1.x_;
        f32 dy = v0.y_ - v1.y_;
        return (dx*dx + dy*dy);
    }

    f32 distance(const Vector2& v0, const Vector2& v1)
    {
        return ::sqrtf(distanceSqr(v0, v1));
    }

    Vector2 lerp(const Vector2& v0, const Vector2& v1, f32 t)
    {
        Vector2 d ={v1.x_-v0.x_, v1.y_-v0.y_};
        d *= t;
        d += v0;
        return lrender::move(d);
    }

    Vector2 muladd(f32 f, const Vector2& v0, const Vector2& v1)
    {
        f32 x = f*v0.x_ + v1.x_;
        f32 y = f*v0.y_ + v1.y_;
        return {x,y};
    }

    Vector2 add(const Vector2& v0, const Vector2& v1)
    {
        return {v0.x_+v1.x_, v0.y_+v1.y_};
    }

    Vector2 sub(const Vector2& v0, const Vector2& v1)
    {
        return {v0.x_-v1.x_, v0.y_-v1.y_};
    }

    Vector2 mul(f32 f, const Vector2& v)
    {
        return {f*v.x_, f*v.y_};
    }

    Vector2 mul(const Vector2& v, f32 f)
    {
        return {v.x_*f, v.y_*f};
    }

    Vector2 muladd(f32 f, const Vector2& v0, const Vector2& v1);

    Vector2 minimum(const Vector2& v0, const Vector2& v1)
    {
        return {lrender::minimum(v0.x_, v1.x_), lrender::minimum(v0.y_, v1.y_)};
    }

    Vector2 maximum(const Vector2& v0, const Vector2& v1)
    {
        return {lrender::maximum(v0.x_, v1.x_), lrender::maximum(v0.y_, v1.y_)};
    }

    f32 minimum(const Vector2& v)
    {
        return lrender::minimum(v.x_, v.y_);
    }

    f32 maximum(const Vector2& v)
    {
        return lrender::maximum(v.x_, v.y_);
    }

    //--------------------------------------------
    //---
    //--- Vector3
    //---
    //--------------------------------------------
    const Vector3 Vector3::Zero(0.0f);
    const Vector3 Vector3::One(1.0f);

    const Vector3 Vector3::Forward ={0.0f, 0.0f, 1.0f};
    const Vector3 Vector3::Backward={0.0f, 0.0f, -1.0f};
    const Vector3 Vector3::Up ={0.0f, 1.0f, 0.0f};
    const Vector3 Vector3::Down ={0.0f, -1.0f, 0.0f};
    const Vector3 Vector3::Right ={1.0f, 0.0f, 0.0f};
    const Vector3 Vector3::Left ={-1.0f, 0.0f, 0.0f};

    Vector3::Vector3(f32 xyz)
        :x_(xyz)
        ,y_(xyz)
        ,z_(xyz)
    {}

    Vector3::Vector3(f32 x, f32 y, f32 z)
        :x_(x)
        ,y_(y)
        ,z_(z)
    {}

    Vector3::Vector3(const Vector4& v)
        :x_(v.x_)
        ,y_(v.y_)
        ,z_(v.z_)
    {}

    Vector3::Vector3(const lm128& v)
    {
        store3(&x_, v);
    }

    void Vector3::zero()
    {
        x_ = y_ = z_ = 0.0f;
    }

    void Vector3::one()
    {
        x_ = y_ = z_ = 1.0f;
    }

    void Vector3::set(f32 x, f32 y, f32 z)
    {
        x_ = x; y_ = y; z_ = z;
    }

    void Vector3::set(const Vector4& v)
    {
        x_ = v.x_; y_ = v.y_; z_ = v.z_;
    }

    Vector3& Vector3::operator+=(const Vector3& v)
    {
        x_ += v.x_;
        y_ += v.y_;
        z_ += v.z_;
        return *this;
    }

    Vector3& Vector3::operator-=(const Vector3& v)
    {
        x_ -= v.x_;
        y_ -= v.y_;
        z_ -= v.z_;
        return *this;
    }

    Vector3& Vector3::operator*=(f32 f)
    {
        x_ *= f;
        y_ *= f;
        z_ *= f;
        return *this;
    }

    Vector3& Vector3::operator/=(f32 f)
    {
        LASSERT(!isZero(f));

#if defined(LMATH_USE_SSE)
        lm128 xv0 = load(*this);
        lm128 xv1 = load(f);
        lm128 xv2 = _mm_div_ps(xv0, xv1);
        store(*this, xv2);

#else
        f = 1.0f / f;
        x_ *= f;
        y_ *= f;
        z_ *= f;
#endif
        return *this;
    }

    Vector3& Vector3::operator*=(const Vector3& v)
    {
        x_ *= v.x_;
        y_ *= v.y_;
        z_ *= v.z_;
        return *this;
    }

    Vector3& Vector3::operator/=(const Vector3& v)
    {
        LASSERT(!isZero(v.x_));
        LASSERT(!isZero(v.y_));
        LASSERT(!isZero(v.z_));

        x_ /= v.x_;
        y_ /= v.y_;
        z_ /= v.z_;
        return *this;
    }

    bool Vector3::isEqual(const Vector3& v) const
    {
        return (lrender::isEqual(x_, v.x_)
            && lrender::isEqual(y_, v.y_)
            && lrender::isEqual(z_, v.z_));
    }

    bool Vector3::isEqual(const Vector3& v, f32 epsilon) const
    {
        return (lrender::isEqual(x_, v.x_, epsilon)
            && lrender::isEqual(y_, v.y_, epsilon)
            && lrender::isEqual(z_, v.z_, epsilon));
    }

    f32 Vector3::lengthSqr() const
    {
        return (x_ * x_ + y_ * y_ + z_ * z_);
    }

    void Vector3::swap(Vector3& rhs)
    {
        lrender::swap(x_, rhs.x_);
        lrender::swap(y_, rhs.y_);
        lrender::swap(z_, rhs.z_);
    }

    bool Vector3::isNan() const
    {
        return (lrender::isNan(x_) || lrender::isNan(y_) || lrender::isNan(z_));
    }

    //--- Vector3's friend functions
    //--------------------------------------------------
    Vector3 operator+(const Vector3& v0, const Vector3& v1)
    {
        return {v0.x_+v1.x_, v0.y_+v1.y_, v0.z_+v1.z_};
    }

    Vector3 operator-(const Vector3& v0, const Vector3& v1)
    {
        return {v0.x_-v1.x_, v0.y_-v1.y_, v0.z_-v1.z_};
    }

    Vector3 operator*(f32 f, const Vector3& v)
    {
        return {f*v.x_, f*v.y_, f*v.z_};
    }

    Vector3 operator*(const Vector3& v, f32 f)
    {
        return {v.x_*f, v.y_*f, v.z_*f};
    }

    Vector3 operator*(const Vector3& v0, const Vector3& v1)
    {
        return {v0.x_*v1.x_, v0.y_*v1.y_, v0.z_*v1.z_};
    }

    Vector3 operator/(const Vector3& v, f32 f)
    {
        LASSERT(!isZero(f));

#if defined(LMATH_USE_SSE)
        lm128 xv0 = load(v);
        lm128 xv1 = load(f);
        lm128 xv2 = _mm_div_ps(xv0, xv1);
        return Vector3(xv2);
#else
        f = 1.0f / f;
        return Vector3(v.x_*f, v.y_*f, v.z_*f);
#endif
    }

    Vector3 operator/(const Vector3& v0, const Vector3& v1)
    {
        LASSERT(!isZero(v1.x_));
        LASSERT(!isZero(v1.y_));
        LASSERT(!isZero(v1.z_));

        lm128 xv0 = load(v0);
        lm128 xv1 = load(v1);
        lm128 xv2 = _mm_div_ps(xv0, xv1);
        return Vector3(xv2);
    }

    Vector3 normalize(const Vector3& v)
    {
        f32 l = v.lengthSqr();
        LASSERT(!isZero(l));

#if defined(LMATH_USE_SSE)
        lm128 xv0 = load(v);
        lm128 xv1 = load(l);
        xv1 = _mm_sqrt_ps(xv1);
        lm128 xv2 = _mm_div_ps(xv0, xv1);
        return Vector3(xv2);
#else
        //l = lcore::rsqrt(l);
        l = 1.0f/ lcore::sqrtf(l);
        return Vector3(v.x_*l, v.y_*l, v.z_*l);
#endif
    }

    Vector3 normalize(const Vector3& v, f32 lengthSqr)
    {
#if defined(LMATH_USE_SSE)
        lm128 xv0 = load(v);
        lm128 xv1 = load(lengthSqr);
        xv1 = _mm_sqrt_ps(xv1);
        lm128 xv2 = _mm_div_ps(xv0, xv1);
        return Vector3(xv2);
#else
        f32 l = 1.0f/ lcore::sqrtf(lengthSqr);
        return Vector3(v.x_*l, v.y_*l, v.z_*l);
#endif
    }

    Vector3 normalizeChecked(const Vector3& v)
    {
        f32 l = v.lengthSqr();
        if(isZeroPositive(l)){
            return Vector3::Zero;
        } else{
            return normalize(v, l);
        }
    }

    Vector3 normalizeChecked(const Vector3& v, const Vector3& default)
    {
        f32 l = v.lengthSqr();
        if(isZeroPositive(l)){
            return default;
        } else{
            return normalize(v, l);
        }
    }

    Vector3 absolute(const Vector3& v)
    {
        return {lrender::absolute(v.x_), lrender::absolute(v.y_), lrender::absolute(v.z_)};
    }

    f32 dot(const Vector3& v0, const Vector3& v1)
    {
        return (v0.x_*v1.x_ + v0.y_*v1.y_ + v0.z_*v1.z_);
    }

    f32 distanceSqr(const Vector3& v0, const Vector3& v1)
    {
        const f32 dx = v0.x_ - v1.x_;
        const f32 dy = v0.y_ - v1.y_;
        const f32 dz = v0.z_ - v1.z_;
        return (dx*dx + dy*dy + dz*dz);
    }

    Vector3 cross(const Vector3& v0, const Vector3& v1)
    {
#if 0
        static const u32 RotMaskR = 201;
        static const u32 RotMaskL = 210;

        lm128 xv0 = Vector3::load(v0.y_, v0.z_, v0.x_);
        lm128 xv1 = Vector3::load(v1.z_, v1.x_, v1.y_);
        lm128 xv2 = _mm_mul_ps(xv0, xv1);

        xv0 = _mm_permute_ps(xv0, RotMaskR);
        xv1 = _mm_permute_ps(xv1, RotMaskL);

        lm128 xv3 = _mm_mul_ps(xv0, xv1);

        xv3 = _mm_sub_ps(xv2, xv3);
        Vector3 v;
        Vector3::store(v, xv3);
        return v;

#else
        f32 x = v0.y_ * v1.z_ - v0.z_ * v1.y_;
        f32 y = v0.z_ * v1.x_ - v0.x_ * v1.z_;
        f32 z = v0.x_ * v1.y_ - v0.y_ * v1.x_;
        return {x,y,z};
#endif
    }

    Vector3 lerp(const Vector3& v0, const Vector3& v1, f32 t)
    {
        Vector3 tmp ={v1.x_-v0.x_, v1.y_-v0.y_, v1.z_-v0.z_};
        tmp.x_ = tmp.x_*t + v0.x_;
        tmp.y_ = tmp.y_*t + v0.y_;
        tmp.z_ = tmp.z_*t + v0.z_;
        return tmp;
    }

    Vector3 lerp(const Vector3& v0, const Vector3& v1, f32 t0, f32 t1)
    {
        Vector3 tmp0 ={v0.x_*t1, v0.y_*t1, v0.z_*t1};
        Vector3 tmp1 ={v1.x_*t0, v1.y_*t0, v1.z_*t0};
        return {tmp0.x_+tmp1.x_, tmp0.y_+tmp1.y_, tmp0.z_+tmp1.z_};
    }


    Vector3 mul(const Matrix34& m, const Vector3& v)
    {
#if defined(LMATH_USE_SSE)
        lm128 tv0 = load(v.x_);
        lm128 tv1 = load(v.y_);
        lm128 tv2 = load(v.z_);

        lm128 tm0 = load(m.m_[0][0], m.m_[1][0], m.m_[2][0]);
        lm128 tm1 = load(m.m_[0][1], m.m_[1][1], m.m_[2][1]);
        lm128 tm2 = load(m.m_[0][2], m.m_[1][2], m.m_[2][2]);
        lm128 tm3 = load(m.m_[0][3], m.m_[1][3], m.m_[2][3]);

        tm0 = _mm_fmadd_ps(tm0, tv0, tm3);
        tm0 = _mm_fmadd_ps(tm1, tv1, tm0);
        tm0 = _mm_fmadd_ps(tm2, tv2, tm0);

        return Vector3(tm0);
#else
        f32 x, y, z;
        x = v.x_ * m.m_[0][0] + v.y_ * m.m_[0][1] + v.z_ * m.m_[0][2] + m.m_[0][3];
        y = v.x_ * m.m_[1][0] + v.y_ * m.m_[1][1] + v.z_ * m.m_[1][2] + m.m_[1][3];
        z = v.x_ * m.m_[2][0] + v.y_ * m.m_[2][1] + v.z_ * m.m_[2][2] + m.m_[2][3];
        return Vector3(x, y, z);
#endif
    }

    Vector3 mul(const Vector3& v, const Matrix34& m)
    {
#if defined(LMATH_USE_SSE)
        lm128 tv0 = load(v.x_);
        lm128 tv1 = load(v.y_);
        lm128 tv2 = load(v.z_);

        lm128 tm0 = _mm_loadu_ps(&m.m_[0][0]);
        lm128 tm1 = _mm_loadu_ps(&m.m_[1][0]);
        lm128 tm2 = _mm_loadu_ps(&m.m_[2][0]);
        lm128 tm3 = load(m.m_[0][3], m.m_[1][3], m.m_[2][3]);

        tm0 = _mm_fmadd_ps(tm0, tv0, tm3);
        tm0 = _mm_fmadd_ps(tm1, tv1, tm0);
        tm0 = _mm_fmadd_ps(tm2, tv2, tm0);

        return Vector3(tm0);
#else
        f32 x, y, z;
        x = v.x_ * m.m_[0][0] + v.y_ * m.m_[1][0] + v.z_ * m.m_[2][0] + m.m_[0][3];
        y = v.x_ * m.m_[0][1] + v.y_ * m.m_[1][1] + v.z_ * m.m_[2][1] + m.m_[1][3];
        z = v.x_ * m.m_[0][2] + v.y_ * m.m_[1][2] + v.z_ * m.m_[2][2] + m.m_[2][3];
        return Vector3(x, y, z);
#endif
    }

    Vector3 mul33(const Matrix34& m, const Vector3& v)
    {
#if defined(LMATH_USE_SSE)
        lm128 tv0 = load(v.x_);
        lm128 tv1 = load(v.y_);
        lm128 tv2 = load(v.z_);

        lm128 tm0 = load(m.m_[0][0], m.m_[1][0], m.m_[2][0]);
        lm128 tm1 = load(m.m_[0][1], m.m_[1][1], m.m_[2][1]);
        lm128 tm2 = load(m.m_[0][2], m.m_[1][2], m.m_[2][2]);

        tm0 = _mm_mul_ps(tm0, tv0);
        tm0 = _mm_fmadd_ps(tm1, tv1, tm0);
        tm0 = _mm_fmadd_ps(tm2, tv2, tm0);

        return Vector3(tm0);

#else
        f32 x, y, z;
        x = v.x_ * m.m_[0][0] + v.y_ * m.m_[0][1] + v.z_ * m.m_[0][2];
        y = v.x_ * m.m_[1][0] + v.y_ * m.m_[1][1] + v.z_ * m.m_[1][2];
        z = v.x_ * m.m_[2][0] + v.y_ * m.m_[2][1] + v.z_ * m.m_[2][2];
        return Vector3(x, y, z);
#endif
    }

    Vector3 mul33(const Vector3& v, const Matrix34& m)
    {
#if defined(LMATH_USE_SSE)
        lm128 tv0 = load(v.x_);
        lm128 tv1 = load(v.y_);
        lm128 tv2 = load(v.z_);

        lm128 tm0 = _mm_loadu_ps(&m.m_[0][0]);
        lm128 tm1 = _mm_loadu_ps(&m.m_[1][0]);
        lm128 tm2 = _mm_loadu_ps(&m.m_[2][0]);

        tm0 = _mm_mul_ps(tm0, tv0);
        tm0 = _mm_fmadd_ps(tm1, tv1, tm0);
        tm0 = _mm_fmadd_ps(tm2, tv2, tm0);

        return Vector3(tm0);

#else
        f32 x, y, z;
        x = v.x_ * m.m_[0][0] + v.y_ * m.m_[1][0] + v.z_ * m.m_[2][0];
        y = v.x_ * m.m_[0][1] + v.y_ * m.m_[1][1] + v.z_ * m.m_[2][1];
        z = v.x_ * m.m_[0][2] + v.y_ * m.m_[1][2] + v.z_ * m.m_[2][2];
        return Vector3(x, y, z);
#endif
    }

    Vector3 mul33(const Matrix44& m, const Vector3& v)
    {
#if defined(LMATH_USE_SSE)
        lm128 tv0 = load(v.x_);
        lm128 tv1 = load(v.y_);
        lm128 tv2 = load(v.z_);

        lm128 tm0 = load(m.m_[0][0], m.m_[1][0], m.m_[2][0]);
        lm128 tm1 = load(m.m_[0][1], m.m_[1][1], m.m_[2][1]);
        lm128 tm2 = load(m.m_[0][2], m.m_[1][2], m.m_[2][2]);

        tm0 = _mm_mul_ps(tm0, tv0);
        tm0 = _mm_fmadd_ps(tm1, tv1, tm0);
        tm0 = _mm_fmadd_ps(tm2, tv2, tm0);

        return Vector3(tm0);

#else
        f32 x, y, z;
        x = v.x_ * m.m_[0][0] + v.y_ * m.m_[0][1] + v.z_ * m.m_[0][2];
        y = v.x_ * m.m_[1][0] + v.y_ * m.m_[1][1] + v.z_ * m.m_[1][2];
        z = v.x_ * m.m_[2][0] + v.y_ * m.m_[2][1] + v.z_ * m.m_[2][2];
        return Vector3(x, y, z);
#endif
    }

    Vector3 mul33(const Vector3& v, const Matrix44& m)
    {
#if defined(LMATH_USE_SSE)
        lm128 tv0 = load(v.x_);
        lm128 tv1 = load(v.y_);
        lm128 tv2 = load(v.z_);

        lm128 tm0 = _mm_loadu_ps(&m.m_[0][0]);
        lm128 tm1 = _mm_loadu_ps(&m.m_[1][0]);
        lm128 tm2 = _mm_loadu_ps(&m.m_[2][0]);

        tm0 = _mm_mul_ps(tm0, tv0);
        tm0 = _mm_fmadd_ps(tm1, tv1, tm0);
        tm0 = _mm_fmadd_ps(tm2, tv2, tm0);
        return Vector3(tm0);

#else
        f32 x, y, z;
        x = v.x_ * m.m_[0][0] + v.y_ * m.m_[1][0] + v.z_ * m.m_[2][0];
        y = v.x_ * m.m_[0][1] + v.y_ * m.m_[1][1] + v.z_ * m.m_[2][1];
        z = v.x_ * m.m_[0][2] + v.y_ * m.m_[1][2] + v.z_ * m.m_[2][2];
        return Vector3(x, y, z);
#endif
    }

    Vector3 rotate(const Vector3& v, const Quaternion& rotation)
    {
        //conjugate(Q) x V x Q
        Quaternion conj = conjugate(rotation);
        Quaternion rot = mul(conj, v);
        rot = mul(rot, rotation);

        return {rot.x_, rot.y_, rot.z_};
    }

    Vector3 rotate(const Quaternion& rotation, const Vector3& v)
    {
        //conjugate(Q) x V x Q
        Quaternion conj = conjugate(rotation);
        Quaternion rot = mul(conj, v);
        rot = mul(rot, rotation);

        return {rot.x_, rot.y_, rot.z_};
    }

    Vector3 mul(const Vector3& v0, const Vector3& v1)
    {
        return {v0.x_ * v1.x_, v0.y_ * v1.y_, v0.z_ * v1.z_};
    }

    Vector3 div(const Vector3& v0, const Vector3& v1)
    {
        LASSERT(!isZero(v1.x_));
        LASSERT(!isZero(v1.y_));
        LASSERT(!isZero(v1.z_));

        lm128 xv0 = load(v0);
        lm128 xv1 = load(v1);
        lm128 xv2 = _mm_div_ps(xv0, xv1);
        return Vector3(xv2);
    }

    Vector3 minimum(const Vector3& v0, const Vector3& v1)
    {
        return {lrender::minimum(v0.x_, v1.x_),
            lrender::minimum(v0.y_, v1.y_),
            lrender::minimum(v0.z_, v1.z_)};
    }

    Vector3 maximum(const Vector3& v0, const Vector3& v1)
    {
        return {lrender::maximum(v0.x_, v1.x_),
            lrender::maximum(v0.y_, v1.y_),
            lrender::maximum(v0.z_, v1.z_)};
    }

    f32 minimum(const Vector3& v)
    {
        return lrender::minimum(lrender::minimum(v.x_, v.y_), v.z_);
    }

    f32 maximum(const Vector3& v)
    {
        return lrender::maximum(lrender::maximum(v.x_, v.y_), v.z_);
    }

    // v0*v1 + v2
    Vector3 muladd(const Vector3& v0, const Vector3& v1, const Vector3& v2)
    {
        lm128 xv0 = load(v0);
        lm128 xv1 = load(v1);
        lm128 xv2 = load(v2);
        lm128 xv3 = _mm_add_ps(_mm_mul_ps(xv0, xv1), xv2);
        return Vector3(xv3);
    }

    // a*v1 + v2
    Vector3 muladd(f32 a, const Vector3& v0, const Vector3& v1)
    {
        lm128 xv0 = load(a);
        lm128 xv1 = load(v0);
        lm128 xv2 = load(v1);
        lm128 xv3 = _mm_add_ps(_mm_mul_ps(xv0, xv1), xv2);
        return Vector3(xv3);
    }

    //--------------------------------------------
    //---
    //--- Vector4
    //---
    //--------------------------------------------
    const Vector4 Vector4::Zero ={0.0f, 0.0f, 0.0f, 0.0f};
    const Vector4 Vector4::One ={1.0f, 1.0f, 1.0f, 1.0f};
    const Vector4 Vector4::Identity ={0.0f, 0.0f, 0.0f, 1.0f};
    const Vector4 Vector4::Forward ={0.0f, 0.0f, 1.0f, 0.0f};
    const Vector4 Vector4::Backward ={0.0f, 0.0f, -1.0f, 0.0f};
    const Vector4 Vector4::Up ={0.0f, 1.0f, 0.0f, 0.0f};
    const Vector4 Vector4::Down ={0.0f, -1.0f, 0.0f, 0.0f};
    const Vector4 Vector4::Right ={1.0f, 0.0f, 0.0f, 0.0f};
    const Vector4 Vector4::Left ={-1.0f, 0.0f, 0.0f, 0.0f};

    Vector4::Vector4(f32 xyzw)
    {
        _mm_storeu_ps(&x_, _mm_set1_ps(xyzw));
    }

    Vector4::Vector4(f32 x, f32 y, f32 z)
        :x_(x)
        ,y_(y)
        ,z_(z)
        ,w_(0.0f)
    {}

    Vector4::Vector4(f32 x, f32 y, f32 z, f32 w)
        :x_(x)
        ,y_(y)
        ,z_(z)
        ,w_(w)
    {}

    void Vector4::zero()
    {
        _mm_storeu_ps(&x_, _mm_setzero_ps());
    }

    void Vector4::one()
    {
        _mm_storeu_ps(&x_, _mm_load_ps(lrender::One));
    }

    void Vector4::identity()
    {
        _mm_storeu_ps(&x_, _mm_loadu_ps(&Identity.x_));
    }

    Vector4::Vector4(const Vector3& v)
        :x_(v.x_)
        ,y_(v.y_)
        ,z_(v.z_)
        ,w_(0.0f)
    {}

    Vector4::Vector4(const Vector3& v, f32 w)
        :x_(v.x_)
        ,y_(v.y_)
        ,z_(v.z_)
        ,w_(w)
    {}

    void Vector4::set(f32 x, f32 y, f32 z, f32 w)
    {
        x_ = x; y_ = y; z_ = z; w_ = w;
    }

    void Vector4::set(f32 v)
    {
#if defined(LMATH_USE_SSE)
        lm128 t = _mm_set1_ps(v);
        store(*this, t);
#else
        x_ = y_ = z_ = w_ = v;
#endif
    }

    void Vector4::set(const Vector3& v)
    {
        x_ = v.x_;
        y_ = v.y_;
        z_ = v.z_;
        w_ = 0.0f;
    }

    void Vector4::set(const Vector3& v, f32 w)
    {
        x_ = v.x_;
        y_ = v.y_;
        z_ = v.z_;
        w_ = w;
    }

    void Vector4::set(const lm128& v)
    {
        store(*this, v);
    }

    Vector4 Vector4::operator-() const
    {
#if defined(LMATH_USE_SSE)
        f32 f;
        *((u32*)&f) = 0x80000000U;
        lm128 mask = _mm_set1_ps(f);
        lm128 r0 = load(*this);
        r0 = _mm_xor_ps(r0, mask);

        Vector4 ret;
        store(ret, r0);
        return lrender::move(ret);
#else
        return Vector4(-x_, -y_, -z_, -w_);
#endif
    }

    Vector4& Vector4::operator+=(const Vector4& v)
    {
#if defined(LMATH_USE_SSE)
        lm128 r0 = load(*this);
        lm128 r1 = load(v);
        r0 = _mm_add_ps(r0, r1);
        store(*this, r0);

#else
        x_ += v.x_;
        y_ += v.y_;
        z_ += v.z_;
        w_ += v.w_;
#endif
        return *this;
    }

    Vector4& Vector4::operator-=(const Vector4& v)
    {
#if defined(LMATH_USE_SSE)
        lm128 r0 = load(*this);
        lm128 r1 = load(v);
        r0 = _mm_sub_ps(r0, r1);
        store(*this, r0);

#else
        x_ -= v.x_;
        y_ -= v.y_;
        z_ -= v.z_;
        w_ -= v.w_;
#endif
        return *this;
    }

    Vector4& Vector4::operator*=(f32 f)
    {
#if defined(LMATH_USE_SSE)
        lm128 r0 = load(*this);
        lm128 r1 = _mm_set1_ps(f);
        r0 = _mm_mul_ps(r0, r1);
        store(*this, r0);

#else
        x_ *= f;
        y_ *= f;
        z_ *= f;
        w_ *= f;
#endif
        return *this;
    }

    Vector4& Vector4::operator/=(f32 f)
    {
#if defined(LMATH_USE_SSE)
        lm128 r0 = load(*this);
        lm128 r1 = _mm_set1_ps(f);
        r0 = _mm_div_ps(r0, r1);
        store(*this, r0);

#else
        LASSERT(!isEqual(f, 0.0f));
        f = 1.0f / f;
        x_ *= f;
        y_ *= f;
        z_ *= f;
        w_ *= f;
#endif
        return *this;
    }

    Vector4& Vector4::operator*=(const Vector4& v)
    {
#if defined(LMATH_USE_SSE)
        lm128 r0 = load(*this);
        lm128 r1 = load(v);
        r0 = _mm_mul_ps(r0, r1);
        store(*this, r0);
#else
        x_ *= v.x_;
        y_ *= v.y_;
        z_ *= v.z_;
        w_ *= v.w_;
#endif
        return *this;
    }

    Vector4& Vector4::operator/=(const Vector4& v)
    {
#if defined(LMATH_USE_SSE)
        lm128 r0 = load(*this);
        lm128 r1 = load(v);
        r0 = _mm_div_ps(r0, r1);
        store(*this, r0);
#else
        x_ /= v.x_;
        y_ /= v.y_;
        z_ /= v.z_;
        w_ /= v.w_;
#endif
        return *this;
    }

    bool Vector4::isEqual(const Vector4& v) const
    {
        return (lrender::isEqual(x_, v.x_)
            && lrender::isEqual(y_, v.y_)
            && lrender::isEqual(z_, v.z_)
            && lrender::isEqual(w_, v.w_));
    }

    bool Vector4::isEqual(const Vector4& v, f32 epsilon) const
    {
        return (lrender::isEqual(x_, v.x_, epsilon)
            && lrender::isEqual(y_, v.y_, epsilon)
            && lrender::isEqual(z_, v.z_, epsilon)
            && lrender::isEqual(w_, v.w_, epsilon));
    }

    void Vector4::setLength()
    {
#if defined(LMATH_USE_SSE)
        lm128 r0 = load(*this);
        r0 = _mm_mul_ps(r0, r0);
        r0 = _mm_add_ps(_mm_permute_ps(r0, 0x4E), r0);
        r0 = _mm_add_ps(_mm_permute_ps(r0, 0xB1), r0);

        r0 = _mm_sqrt_ss(r0);
        r0 = _mm_permute_ps(r0, 0x00);
        store(*this, r0);
#else
        x_ = x_ * x_;
        y_ = y_ * y_;
        z_ = z_ * z_;
        w_ = w_ * w_;
#endif
    }

    f32 Vector4::length() const
    {
#if defined(LMATH_USE_SSE)
        lm128 r0 = load(*this);
        r0 = _mm_mul_ps(r0, r0);
        r0 = _mm_add_ps(_mm_permute_ps(r0, 0x4E), r0);
        r0 = _mm_add_ps(_mm_permute_ps(r0, 0xB1), r0);

        r0 = _mm_sqrt_ss(r0);
        f32 ret;
        _mm_store_ss(&ret, r0);
        return ret;
#else
        return lcore::sqrtf(lengthSqr());
#endif
    }

    f32 Vector4::lengthSqr() const
    {
        return (x_ * x_ + y_ * y_ + z_ * z_ + w_ * w_);
    }

    void Vector4::swap(Vector4& rhs)
    {
#if defined(LMATH_USE_SSE)
        lm128 t0 = load(*this);
        lm128 t1 = load(rhs);

        store(*this, t1);
        store(rhs, t0);
#else
        lcore::swap(x_, rhs.x_);
        lcore::swap(y_, rhs.y_);
        lcore::swap(z_, rhs.z_);
        lcore::swap(w_, rhs.w_);
#endif
    }

    bool Vector4::isNan() const
    {
        return (lrender::isNan(x_) || lrender::isNan(y_) || lrender::isNan(z_) || lrender::isNan(w_));
    }

    bool Vector4::isZero() const
    {
        return (lrender::isZero(x_) && lrender::isZero(y_) && lrender::isZero(z_) && lrender::isZero(w_));
    }

    //--- Vector4's friend functions
    //--------------------------------------------------
    void copy(Vector4& dst, const Vector4& src)
    {
        _mm_storeu_ps(&dst.x_, _mm_loadu_ps(&src.x_));
    }

    Vector4 normalize(const Vector4& v)
    {
#if defined(LMATH_USE_SSE)
        lm128 r0 = load(v);
        lm128 r1 = r0;
        r0 = _mm_mul_ps(r0, r0);
        r0 = _mm_add_ps(_mm_permute_ps(r0, 0x4E), r0);
        r0 = _mm_add_ps(_mm_permute_ps(r0, 0xB1), r0);

        r0 = _mm_sqrt_ss(r0);
        r0 = _mm_permute_ps(r0, 0);
#if 0
        r0 = rcp(r0);
        r1 = _mm_mul_ps(r1, r0);
#else
        r1 = _mm_div_ps(r1, r0);
#endif
        return Vector4(r1);
#else
        f32 l = lengthSqr();
        LASSERT(!lcore::isZero(l));
        l = 1.0f/ lcore::sqrtf(l);
        return Vector4(v.x_*l, v.y_*l, v.z_*l, v.w_*l);
#endif
    }

    Vector4 normalize(const Vector4& v, f32 lengthSqr)
    {
#if defined(LMATH_USE_SSE)
        lm128 r0 = _mm_set1_ps(lengthSqr);
        lm128 r1 = load(v);
        r0 = _mm_sqrt_ps(r0);

#if 0
        r0 = rcp(r0);
        r1 = _mm_mul_ps(r1, r0);
#else
        r1 = _mm_div_ps(r1, r0);
#endif
        return Vector4(r1);
#else
        f32 l = lengthSqr;
        LASSERT(!(isEqual(l, 0.0f)));
        //l = lcore::rsqrt(l);
        l = 1.0f/ lcore::sqrtf(l);
        return Vector4(v.x_*l, v.y_*l, v.z_*l, v.w_*l);
#endif
    }

    Vector4 normalizeChecked(const Vector4& v)
    {
#if defined(LMATH_USE_SSE)
        f32 l = v.lengthSqr();
        if(isZeroPositive(l)){
            return Vector4(_mm_setzero_ps());
        } else{
            return normalize(v, l);
        }
#else
        f32 l = v.lengthSqr();
        if(lcore::isZeroPositive(l)){
            return Vector4(0.0f);
        } else{
            return normalize(v, l);
        }
#endif
    }

    Vector4 absolute(const Vector4& v)
    {
        return Vector4(_mm_andnot_ps(_mm_set1_ps(-0.0f), load(v)));
    }

    f32 dot(const Vector4& v0, const Vector4& v1)
    {
#if defined(LMATH_USE_SSE)
        lm128 r0 = load(v0);
        lm128 r1 = load(v1);
        r0 = _mm_mul_ps(r0, r1);
        r0 = _mm_add_ps(_mm_permute_ps(r0, 0x4E), r0);
        r0 = _mm_add_ps(_mm_permute_ps(r0, 0xB1), r0);

        f32 ret;
        _mm_store_ss(&ret, r0);
        return ret;
#else
        return (x_ * v.x_ + y_ * v.y_ + z_ * v.z_ + w_ * v.w_);
#endif
    }

    Vector4 cross3(const Vector4& v0, const Vector4& v1)
    {
#if defined(LMATH_USE_SSE)
        lm128 xv0 = load(v0);
        lm128 xv1 = load(v1);
        return Vector4(::cross3(xv0, xv1));
#else
        f32 x = v0.y_ * v1.z_ - v0.z_ * v1.y_;
        f32 y = v0.z_ * v1.x_ - v0.x_ * v1.z_;
        f32 z = v0.x_ * v1.y_ - v0.y_ * v1.x_;
        return Vector4(x, y, z, 0.0f);
#endif
    }

    f32 distanceSqr(const Vector4& v0, const Vector4& v1)
    {
#if defined(LMATH_USE_SSE)
        lm128 r0 = load(v0);
        lm128 r1 = load(v1);
        lm128 d = _mm_sub_ps(r0, r1);

        d = _mm_mul_ps(d, d);
        d = _mm_add_ps(_mm_permute_ps(d, 0x4E), d);
        d = _mm_add_ps(_mm_permute_ps(d, 0xB1), d);

        f32 ret;
        _mm_store_ss(&ret, d);
        return ret;

#else
        const f32 dx = x_ - v.x_;
        const f32 dy = y_ - v.y_;
        const f32 dz = z_ - v.z_;
        const f32 dw = w_ - v.w_;
        return (dx * dx + dy * dy + dz * dz + dw * dw);
#endif
    }

    f32 distance(const Vector4& v0, const Vector4& v1)
    {
#if defined(LMATH_USE_SSE)
        lm128 r0 = load(v0);
        lm128 r1 = load(v1);
        lm128 d = _mm_sub_ps(r0, r1);

        d = _mm_mul_ps(d, d);
        d = _mm_add_ps(_mm_permute_ps(d, 0x4E), d);
        d = _mm_add_ps(_mm_permute_ps(d, 0xB1), d);

        d = _mm_sqrt_ss(d);

        f32 ret;
        _mm_store_ss(&ret, d);
        return ret;

#else
        return lcore::sqrtf(distanceSqr(v));
#endif
    }
    f32 manhattanDistance(const Vector4& v0, const Vector4& v1)
    {
#if defined(LMATH_USE_SSE)
        lm128 r0 = load(v0);
        lm128 r1 = load(v1);
        lm128 d = _mm_sub_ps(r0, r1);
        f32 f;
        *((u32*)&f) = 0x7FFFFFFFU;
        lm128 m = _mm_set1_ps(f);
        d = _mm_and_ps(d, m);

        d = _mm_add_ps(_mm_permute_ps(d, 0x4E), d);
        d = _mm_add_ps(_mm_permute_ps(d, 0xB1), d);

        f32 ret;
        _mm_store_ss(&ret, d);
        return ret;
#else
        Vector4 tmp;
        tmp.sub(*this, v);
        return lcore::absolute(tmp.x_) + lcore::absolute(tmp.y_) + lcore::absolute(tmp.z_) + lcore::absolute(tmp.w_);
#endif
    }

    f32 distanceSqr3(const Vector3& v0, const Vector4& v1)
    {
        const f32 dx = v0.x_ - v1.x_;
        const f32 dy = v0.y_ - v1.y_;
        const f32 dz = v0.z_ - v1.z_;
        return (dx * dx + dy * dy + dz * dz);
    }

    f32 distanceSqr3(const Vector4& v0, const Vector4& v1)
    {
        const f32 dx = v0.x_ - v1.x_;
        const f32 dy = v0.y_ - v1.y_;
        const f32 dz = v0.z_ - v1.z_;
        return (dx * dx + dy * dy + dz * dz);
    }

    f32 distance3(const Vector3& v0, const Vector4& v1)
    {
        const f32 dx = v0.x_ - v1.x_;
        const f32 dy = v0.y_ - v1.y_;
        const f32 dz = v0.z_ - v1.z_;
        return ::sqrtf(dx * dx + dy * dy + dz * dz);
    }

    f32 distance3(const Vector4& v0, const Vector4& v1)
    {
        const f32 dx = v0.x_ - v1.x_;
        const f32 dy = v0.y_ - v1.y_;
        const f32 dz = v0.z_ - v1.z_;
        return ::sqrtf(dx * dx + dy * dy + dz * dz);
    }

    f32 manhattanDistance3(const Vector3& v0, const Vector4& v1)
    {
        const f32 dx = v0.x_ - v1.x_;
        const f32 dy = v0.y_ - v1.y_;
        const f32 dz = v0.z_ - v1.z_;;
        return absolute(dx) + absolute(dy) + absolute(dz);
    }

    f32 manhattanDistance3(const Vector4& v0, const Vector4& v1)
    {
#if defined(LMATH_USE_SSE)
        lm128 r0 = load(v0);
        lm128 r1 = load(v1);
        lm128 d = _mm_sub_ps(r0, r1);
        f32 f;
        *((u32*)&f) = 0x7FFFFFFFU;
        lm128 m = _mm_set1_ps(f);
        d = _mm_and_ps(d, m);

        GFX_ALIGN16 f32 tmp[4];
        _mm_store_ps(tmp, d);
        return tmp[0] + tmp[1] + tmp[2];
#else
        const f32 dx = v0.x_ - v1.x_;
        const f32 dy = v0.y_ - v1.y_;
        const f32 dz = v0.z_ - v1.z_;;
        return lcore::absolute(dx) + lcore::absolute(dy) + lcore::absolute(dz);
#endif
    }

    Vector4 mul(const Matrix34& m, const Vector4& v)
    {
#if defined(LMATH_USE_SSE)
        lm128 tm0 = _mm_loadu_ps(&m.m_[0][0]);
        lm128 tm1 = _mm_loadu_ps(&m.m_[1][0]);
        lm128 tm2 = _mm_loadu_ps(&m.m_[2][0]);
        lm128 tm3 = _mm_set_ps(1.0f, 0.0f, 0.0f, 0.0f);

        _MM_TRANSPOSE4_PS(tm0, tm1, tm2, tm3);

        lm128 tv = _mm_loadu_ps(&v.x_);
        lm128 tv0 = _mm_permute_ps(tv, _MM_SHUFFLE(0,0,0,0));
        lm128 tv1 = _mm_permute_ps(tv, _MM_SHUFFLE(1,1,1,1));
        lm128 tv2 = _mm_permute_ps(tv, _MM_SHUFFLE(2,2,2,2));
        lm128 tv3 = _mm_permute_ps(tv, _MM_SHUFFLE(3,3,3,3));

        tm0 = _mm_mul_ps(tm0, tv0);
        tm0 = _mm_fmadd_ps(tm1, tv1, tm0);
        tm0 = _mm_fmadd_ps(tm2, tv2, tm0);
        tm0 = _mm_fmadd_ps(tm3, tv3, tm0);

        return Vector4(tm0);
#else
        f32 x, y, z;
        x = v.x_ * m.m_[0][0] + v.y_ * m.m_[0][1] + v.z_ * m.m_[0][2] + v.w_ * m.m_[0][3];
        y = v.x_ * m.m_[1][0] + v.y_ * m.m_[1][1] + v.z_ * m.m_[1][2] + v.w_ * m.m_[1][3];
        z = v.x_ * m.m_[2][0] + v.y_ * m.m_[2][1] + v.z_ * m.m_[2][2] + v.w_ * m.m_[2][3];
        return Vector4(x, y, z, v.w_);
#endif
    }

    Vector4 mul(const Vector4& v, const Matrix34& m)
    {
#if defined(LMATH_USE_SSE)
        lm128 tv = _mm_loadu_ps(&v.x_);
        lm128 tv0 = _mm_permute_ps(tv, _MM_SHUFFLE(0,0,0,0));
        lm128 tv1 = _mm_permute_ps(tv, _MM_SHUFFLE(1,1,1,1));
        lm128 tv2 = _mm_permute_ps(tv, _MM_SHUFFLE(2,2,2,2));
        lm128 tv3 = _mm_permute_ps(tv, _MM_SHUFFLE(3,3,3,3));

        lm128 tm0 = _mm_loadu_ps(&m.m_[0][0]);
        lm128 tm1 = _mm_loadu_ps(&m.m_[1][0]);
        lm128 tm2 = _mm_loadu_ps(&m.m_[2][0]);
#if 0
        const f32 one = 1.0f;
        lm128 tm3 = _mm_load_ss(&one);
        tm3 = _mm_permute_ps(tm3, 0x2F);
#else
        f32 buffer[4] ={0.0f, 0.0f, 0.0f, 1.0f};
        lm128 tm3 = _mm_loadu_ps(buffer);
#endif

        tm0 = _mm_mul_ps(tm0, tv0);
        tm0 = _mm_fmadd_ps(tm1, tv1, tm0);
        tm0 = _mm_fmadd_ps(tm2, tv2, tm0);
        tm0 = _mm_fmadd_ps(tm3, tv3, tm0);

        return Vector4(tm0);

#else
        f32 x, y, z, w;
        x = v.x_ * m.m_[0][0] + v.y_ * m.m_[1][0] + v.z_ * m.m_[2][0];
        y = v.x_ * m.m_[0][1] + v.y_ * m.m_[1][1] + v.z_ * m.m_[2][1];
        z = v.x_ * m.m_[0][2] + v.y_ * m.m_[1][2] + v.z_ * m.m_[2][2];
        w = v.x_ * m.m_[0][3] + v.y_ * m.m_[1][3] + v.z_ * m.m_[2][3] + v.w_;

        return Vector4(x, y, z, w);
#endif
    }



    Vector4 mul(const Matrix44& m, const Vector4& v)
    {
#if defined(LMATH_USE_SSE)
        lm128 tm0 = _mm_loadu_ps(&m.m_[0][0]);
        lm128 tm1 = _mm_loadu_ps(&m.m_[1][0]);
        lm128 tm2 = _mm_loadu_ps(&m.m_[2][0]);
        lm128 tm3 = _mm_loadu_ps(&m.m_[3][0]);

        _MM_TRANSPOSE4_PS(tm0, tm1, tm2, tm3);

        lm128 tv = _mm_loadu_ps(&v.x_);
        lm128 tv0 = _mm_permute_ps(tv, _MM_SHUFFLE(0,0,0,0));
        lm128 tv1 = _mm_permute_ps(tv, _MM_SHUFFLE(1,1,1,1));
        lm128 tv2 = _mm_permute_ps(tv, _MM_SHUFFLE(2,2,2,2));
        lm128 tv3 = _mm_permute_ps(tv, _MM_SHUFFLE(3,3,3,3));

        tm0 = _mm_mul_ps(tm0, tv0);
        tm0 = _mm_fmadd_ps(tm1, tv1, tm0);
        tm0 = _mm_fmadd_ps(tm2, tv2, tm0);
        tm0 = _mm_fmadd_ps(tm3, tv3, tm0);

        return Vector4(tm0);

#elif defined(LMATH_USE_SSE)
        lm128 tm0 = _mm_loadu_ps(&m.m_[0][0]);
        lm128 tm1 = _mm_loadu_ps(&m.m_[1][0]);
        lm128 tm2 = _mm_loadu_ps(&m.m_[2][0]);
        lm128 tm3 = _mm_loadu_ps(&m.m_[3][0]);
        lm128 tv = _mm_loadu_ps(&v.x_);

        LALIGN16 f32 tmp[4];
        _mm_store_ps(tmp, _mm_mul_ps(tv, tm0));
        x_ = tmp[0] + tmp[1] + tmp[2] + tmp[3];

        _mm_store_ps(tmp, _mm_mul_ps(tv, tm1));
        y_ = tmp[0] + tmp[1] + tmp[2] + tmp[3];

        _mm_store_ps(tmp, _mm_mul_ps(tv, tm2));
        z_ = tmp[0] + tmp[1] + tmp[2] + tmp[3];

        _mm_store_ps(tmp, _mm_mul_ps(tv, tm3));
        w_ = tmp[0] + tmp[1] + tmp[2] + tmp[3];
#else
        f32 x, y, z, w;
        x = v.x_ * m.m_[0][0] + v.y_ * m.m_[0][1] + v.z_ * m.m_[0][2] + v.w_ * m.m_[0][3];
        y = v.x_ * m.m_[1][0] + v.y_ * m.m_[1][1] + v.z_ * m.m_[1][2] + v.w_ * m.m_[1][3];
        z = v.x_ * m.m_[2][0] + v.y_ * m.m_[2][1] + v.z_ * m.m_[2][2] + v.w_ * m.m_[2][3];
        w = v.x_ * m.m_[3][0] + v.y_ * m.m_[3][1] + v.z_ * m.m_[3][2] + v.w_ * m.m_[3][3];
        return Vector4(x, y, z, w);
#endif
    }

    Vector4 mul(const Vector4& v, const Matrix44& m)
    {
#if defined(LMATH_USE_SSE)
        lm128 tv = _mm_loadu_ps(&v.x_);
        lm128 tv0 = _mm_permute_ps(tv, _MM_SHUFFLE(0,0,0,0));
        lm128 tv1 = _mm_permute_ps(tv, _MM_SHUFFLE(1,1,1,1));
        lm128 tv2 = _mm_permute_ps(tv, _MM_SHUFFLE(2,2,2,2));
        lm128 tv3 = _mm_permute_ps(tv, _MM_SHUFFLE(3,3,3,3));

        lm128 tm0 = _mm_loadu_ps(&m.m_[0][0]);
        lm128 tm1 = _mm_loadu_ps(&m.m_[1][0]);
        lm128 tm2 = _mm_loadu_ps(&m.m_[2][0]);
        lm128 tm3 = _mm_loadu_ps(&m.m_[3][0]);

        tm0 = _mm_mul_ps(tm0, tv0);
        tm0 = _mm_fmadd_ps(tm1, tv1, tm0);
        tm0 = _mm_fmadd_ps(tm2, tv2, tm0);
        tm0 = _mm_fmadd_ps(tm3, tv3, tm0);
        return Vector4(tm0);

#else
        f32 x, y, z, w;
        x = v.x_ * m.m_[0][0] + v.y_ * m.m_[1][0] + v.z_ * m.m_[2][0] + v.w_ * m.m_[3][0];
        y = v.x_ * m.m_[0][1] + v.y_ * m.m_[1][1] + v.z_ * m.m_[2][1] + v.w_ * m.m_[3][1];
        z = v.x_ * m.m_[0][2] + v.y_ * m.m_[1][2] + v.z_ * m.m_[2][2] + v.w_ * m.m_[3][2];
        w = v.x_ * m.m_[0][3] + v.y_ * m.m_[1][3] + v.z_ * m.m_[2][3] + v.w_ * m.m_[3][3];
        return Vector4(x, y, z, w);
#endif
    }

    lm128 mul(const lm128& m0, const lm128& m1, const lm128& m2, const lm128& m3,
        const lm128& tv0, const lm128& tv1, const lm128& tv2, const lm128& tv3)
    {
        lm128 result = _mm_mul_ps(m0, tv0);
        result = _mm_fmadd_ps(m1, tv1, result);
        result = _mm_fmadd_ps(m2, tv2, result);
        result = _mm_fmadd_ps(m3, tv3, result);
        return result;
    }

    lm128 mul(const lm128& m0, const lm128& m1, const lm128& m2, const lm128& m3,
        const lm128& tv)
    {
        lm128 tv0 = _mm_permute_ps(tv, _MM_SHUFFLE(0,0,0,0));
        lm128 tv1 = _mm_permute_ps(tv, _MM_SHUFFLE(1,1,1,1));
        lm128 tv2 = _mm_permute_ps(tv, _MM_SHUFFLE(2,2,2,2));
        lm128 tv3 = _mm_permute_ps(tv, _MM_SHUFFLE(3,3,3,3));
        return mul(m0, m1, m2, m3, tv0, tv1, tv2, tv3);
    }

    Vector4 mulPoint(const Matrix44& m, const Vector4& v)
    {
        Vector4 tv=v;
        tv.w_ = 1.0f;

        lm128 tm0 = _mm_loadu_ps(&m.m_[0][0]);
        lm128 tm1 = _mm_loadu_ps(&m.m_[1][0]);
        lm128 tm2 = _mm_loadu_ps(&m.m_[2][0]);
        lm128 tm3 = _mm_loadu_ps(&m.m_[3][0]);

        _MM_TRANSPOSE4_PS(tm0, tm1, tm2, tm3);

        lm128 ttv = _mm_loadu_ps(&tv.x_);
        lm128 tv0 = _mm_permute_ps(ttv, _MM_SHUFFLE(0,0,0,0));
        lm128 tv1 = _mm_permute_ps(ttv, _MM_SHUFFLE(1,1,1,1));
        lm128 tv2 = _mm_permute_ps(ttv, _MM_SHUFFLE(2,2,2,2));
        lm128 tv3 = _mm_permute_ps(ttv, _MM_SHUFFLE(3,3,3,3));

        lm128 t0 = mul(tm0, tm1, tm2, tm3, tv0, tv1, tv2, tv3);
        t0 = _mm_div_ps(t0, _mm_permute_ps(t0, _MM_SHUFFLE(3, 3, 3, 3)));
        store(tv, t0);
        tv.w_ = v.w_;
        return tv;
    }

    Vector4 mulVector(const Matrix44& m, const Vector4& v)
    {
        Vector4 tv=v;
        tv.w_ = 0.0f;

        lm128 tm0 = _mm_loadu_ps(&m.m_[0][0]);
        lm128 tm1 = _mm_loadu_ps(&m.m_[1][0]);
        lm128 tm2 = _mm_loadu_ps(&m.m_[2][0]);
        lm128 tm3 = _mm_loadu_ps(&m.m_[3][0]);

        _MM_TRANSPOSE4_PS(tm0, tm1, tm2, tm3);

        lm128 ttv = _mm_loadu_ps(&tv.x_);
        lm128 tv0 = _mm_permute_ps(ttv, _MM_SHUFFLE(0,0,0,0));
        lm128 tv1 = _mm_permute_ps(ttv, _MM_SHUFFLE(1,1,1,1));
        lm128 tv2 = _mm_permute_ps(ttv, _MM_SHUFFLE(2,2,2,2));
        lm128 tv3 = _mm_permute_ps(ttv, _MM_SHUFFLE(3,3,3,3));

        lm128 t0 = mul(tm0, tm1, tm2, tm3, tv0, tv1, tv2, tv3);
        store(tv, t0);
        tv.w_ = v.w_;
        return tv;
    }

    Vector4 rotate(const Vector4& v, const Quaternion& rotation)
    {
        Quaternion conj = conjugate(rotation);
        Quaternion rot = mul(conj, v);
        rot = mul(rot, rotation);
        return {rot.x_, rot.y_, rot.z_, v.w_};
    }

    Vector4 rotate(const Quaternion& rotation, const Vector4& v)
    {
        Quaternion conj = conjugate(rotation);
        Quaternion rot = mul(conj, v);
        rot = mul(rot, rotation);
        return {rot.x_, rot.y_, rot.z_, v.w_};
    }

    Vector4 operator+(const Vector4& v0, const Vector4& v1)
    {
#if defined(LMATH_USE_SSE)
        lm128 r0 = load(v0);
        lm128 r1 = load(v1);
        return Vector4(_mm_add_ps(r0, r1));

#else
        f32 x = v0.x_ + v1.x_;
        f32 y = v0.y_ + v1.y_;
        f32 z = v0.z_ + v1.z_;
        f32 w = v0.w_ + v1.w_;
        return Vector4(x, y, z, w);
#endif
    }

    Vector4 operator-(const Vector4& v0, const Vector4& v1)
    {
#if defined(LMATH_USE_SSE)
        lm128 r0 = load(v0);
        lm128 r1 = load(v1);
        return Vector4(_mm_sub_ps(r0, r1));

#else
        f32 x = v0.x_ - v1.x_;
        f32 y = v0.y_ - v1.y_;
        f32 z = v0.z_ - v1.z_;
        f32 w = v0.w_ - v1.w_;
        return Vector4(x, y, z, w);
#endif
    }

    Vector4 operator*(f32 f, const Vector4& v)
    {
#if defined(LMATH_USE_SSE)
        lm128 r0 = _mm_set1_ps(f);
        lm128 r1 = load(v);
        return Vector4(_mm_mul_ps(r0, r1));
#else
        f32 x = f * v.x_;
        f32 y = f * v.y_;
        f32 z = f * v.z_;
        f32 w = f * v.w_;
        return Vector4(x, y, z, w);
#endif
    }

    Vector4 operator*(const Vector4& v0, const Vector4& v1)
    {
#if defined(LMATH_USE_SSE)
        lm128 r0 = load(v0);
        lm128 r1 = load(v1);
        return Vector4(_mm_mul_ps(r0, r1));
#else
        return {v0.x_*v1.x_, v0.y_*v1.y_, v0.z_*v1.z_, v0.w_*v1.w_};
#endif
    }

    Vector4 operator/(const Vector4& v, f32 f)
    {
#if defined(LMATH_USE_SSE)
        lm128 r0 = load(v);
        lm128 r1 = _mm_set1_ps(f);
        return Vector4(_mm_div_ps(r0, r1));
#else
        f = 1.0f/f;
        return v*f;
#endif
    }

    Vector4 operator/(const Vector4& v0, const Vector4& v1)
    {
#if defined(LMATH_USE_SSE)
        lm128 r0 = load(v0);
        lm128 r1 = load(v1);
        return Vector4(_mm_div_ps(r0, r1));
#else
        return {v0.x_/v1.x_, v0.y_/v1.y_, v0.z_/v1.z_, v0.w_/v1.w_};
#endif
    }

    Vector4 mul(const Vector4& v0, const Vector4& v1)
    {
#if defined(LMATH_USE_SSE)
        lm128 r0 = load(v0);
        lm128 r1 = load(v1);
        return Vector4(_mm_mul_ps(r0, r1));
#else
        f32 x = v0.x_ * v1.x_;
        f32 y = v0.y_ * v1.y_;
        f32 z = v0.z_ * v1.z_;
        f32 w = v0.w_ * v1.w_;
        return {x,y,z,w};
#endif
    }

    Vector4 div(const Vector4& v0, const Vector4& v1)
    {
#if defined(LMATH_USE_SSE)
        lm128 r0 = load(v0);
        lm128 r1 = load(v1);
        return Vector4(_mm_div_ps(r0, r1));
#else
        f32 x = v0.x_ / v1.x_;
        f32 y = v0.y_ / v1.y_;
        f32 z = v0.z_ / v1.z_;
        f32 w = v0.w_ / v1.w_;
        return {x,y,z,w};
#endif
    }

    Vector4 add(const Vector4& v, f32 f)
    {
#if defined(LMATH_USE_SSE)
        lm128 r0 = load(v);
        lm128 r1 = _mm_set1_ps(f);
        return Vector4(_mm_add_ps(r0, r1));
#else
        return {v.x_+f, v.y_+f, v.z_+f, v.w_+f};
#endif
    }

    Vector4 sub(const Vector4& v, f32 f)
    {
#if defined(LMATH_USE_SSE)
        lm128 r0 = load(v);
        lm128 r1 = _mm_set1_ps(f);
        return Vector4(_mm_sub_ps(r0, r1));
#else
        return {v.x_-f, v.y_-f, v.z_-f, v.w_0f};
#endif
    }

    Vector4 minimum(const Vector4& v0, const Vector4& v1)
    {
#if defined(LMATH_USE_SSE)
        lm128 r0 = load(v0);
        lm128 r1 = load(v1);
        return Vector4(_mm_min_ps(r0, r1));

#else
        f32 x = lcore::minimum(v0.x_, v1.x_);
        f32 y = lcore::minimum(v0.y_, v1.y_);
        f32 z = lcore::minimum(v0.z_, v1.z_);
        f32 w = lcore::minimum(v0.w_, v1.w_);
        return {x,y,z,w};
#endif
    }

    Vector4 maximum(const Vector4& v0, const Vector4& v1)
    {
#if defined(LMATH_USE_SSE)
        lm128 r0 = load(v0);
        lm128 r1 = load(v1);
        return Vector4(_mm_max_ps(r0, r1));

#else
        f32 x = lcore::maximum(v0.x_, v1.x_);
        f32 y = lcore::maximum(v0.y_, v1.y_);
        f32 z = lcore::maximum(v0.z_, v1.z_);
        f32 w = lcore::maximum(v0.w_, v1.w_);
        return {x,y,z,w};
#endif
    }

    f32 minimum(const Vector4& v)
    {
        return lrender::minimum(lrender::minimum(v.x_, v.y_), lrender::minimum(v.z_, v.w_));
    }

    f32 maximum(const Vector4& v)
    {
        return lrender::maximum(lrender::maximum(v.x_, v.y_), lrender::maximum(v.z_, v.w_));
    }

    /**
    @brief v0*v1 + v2
    */
    Vector4 muladd(const Vector4& v0, const Vector4& v1, const Vector4& v2)
    {
#if defined(LMATH_USE_SSE)
        lm128 r0 = load(v0);
        lm128 r1 = load(v1);
        lm128 r2 = load(v2);
        r0 = _mm_mul_ps(r0, r1);
        r0 = _mm_add_ps(r0, r2);
        return Vector4(r0);

#else
        f32 x = v0.x_ * v1.x_ + v2.x_;
        f32 y = v0.y_ * v1.y_ + v2.y_;
        f32 z = v0.z_ * v1.z_ + v2.z_;
        f32 w = v0.w_ * v1.w_ + v2.w_;
        return {x,y,z,w};
#endif
    }

    /**
    @brief a*v0 + v1
    */
    Vector4 muladd(f32 a, const Vector4& v0, const Vector4& v1)
    {
#if defined(LMATH_USE_SSE)
        lm128 r0 = _mm_set1_ps(a);
        lm128 r1 = load(v0);
        lm128 r2 = load(v1);
        r0 = _mm_mul_ps(r0, r1);
        r0 = _mm_add_ps(r0, r2);
        return Vector4(r0);

#else
        f32 x = a * v0.x_ + v1.x_;
        f32 y = a * v0.y_ + v1.y_;
        f32 z = a * v0.z_ + v1.z_;
        f32 w = a * v0.w_ + v1.w_;
        return {x,y,z,w};
#endif
    }

    Vector4 floor(const Vector4& v)
    {
        lm128 tv0 = load(v);
        lm128 tv1 = _mm_cvtepi32_ps(_mm_cvttps_epi32(tv0));

        return Vector4(_mm_sub_ps(tv1, _mm_and_ps(_mm_cmplt_ps(tv0, tv1), _mm_set1_ps(1.0f))));
    }

    Vector4 ceil(const Vector4& v)
    {
        lm128 tv0 = load(v);
        lm128 tv1 = _mm_cvtepi32_ps(_mm_cvttps_epi32(tv0));

        return Vector4(_mm_add_ps(tv1, _mm_and_ps(_mm_cmplt_ps(tv0, tv1), _mm_set1_ps(1.0f))));
    }

    Vector4 invert(const Vector4& v)
    {
#if defined(LMATH_USE_SSE)
        lm128 r0 = _mm_set1_ps(1.0f);
        lm128 r1 = load(v);
        return Vector4(_mm_div_ps(r0, r1));
#else
        f32 x = 1.0f/x_;
        f32 y = 1.0f/y_;
        f32 z = 1.0f/z_;
        f32 w = 1.0f/w_;
        return {x,y,z,w};
#endif
    }

    Vector4 sqrt(const Vector4& v)
    {
#if defined(LMATH_USE_SSE)
        lm128 r0 = load(v);
        return Vector4(_mm_sqrt_ps(r0));
#else
        f32 x = lcore::sqrtf(x_);
        f32 y = lcore::sqrtf(y_);
        f32 z = lcore::sqrtf(z_);
        f32 w = lcore::sqrtf(w_);
        return {x,y,z,w};
#endif
    }

    Vector4 lerp(const Vector4& v0, const Vector4& v1, f32 t)
    {
#if defined(LMATH_USE_SSE)
        lm128 tv0 = load(v0);
        lm128 tv1 = load(v1);
        lm128 t0 = _mm_set1_ps(t);

        tv1 = _mm_sub_ps(tv1, tv0);
        tv1 = _mm_mul_ps(tv1, t0);
        tv1 = _mm_add_ps(tv1, tv0);
        return Vector4(tv1);

#else
        Vector4 tmp v1-v0;
        tmp *= t;
        return tmp+v0;
#endif
    }

    Vector4 slerp(const Vector4& v0, const Vector4& v1, f32 t)
    {
        f32 cosine = dot(v0, v1);
        if(LRENDER_ANGLE_LIMIT1<=absolute(cosine)){
            return lerp(v0, v1, t);
        } else{
            return slerp(v0, v1, t, cosine);
        }
    }

    Vector4 slerp(const Vector4& v0, const Vector4& v1, f32 t, f32 cosine)
    {
        LASSERT(cosine<LRENDER_ANGLE_LIMIT1);

        f32 omega = acosf(cosine);

        f32 inv = 1.0f/::sqrtf(1.0f-cosine*cosine);
        f32 s0 = ::sinf((1.0f-t)*omega) * inv;
        f32 s1 = ::sinf(t*omega) * inv;

        lm128 tv0 = load(v0);
        lm128 tv1 = load(v1);
        lm128 t0 = _mm_set1_ps(s0);
        lm128 t1 = _mm_set1_ps(s1);

        tv0 = _mm_mul_ps(t0, tv0);
        tv1 = _mm_mul_ps(t1, tv1);

        return Vector4(_mm_add_ps(tv0, tv1));
    }

    Vector4 getParallelComponent(const Vector4& v, const Vector4& basis)
    {
        f32 cs = dot(v, basis);
        return (basis*cs);
    }

    Vector4 getPerpendicularComponent(const Vector4& v, const Vector4& basis)
    {
        return (v - getParallelComponent(v, basis));
    }

    Vector4 getLinearZParameterReverseZ(f32 znear, f32 zfar)
    {
        f32 D = znear - zfar;
        f32 invD = 1.0f/D;
        return Vector4(D, znear, -zfar*znear*invD, -zfar*invD);
    }

    f32 toLinearZ(f32 z, const Vector4& parameter)
    {
        return parameter.z_/(parameter.x_*z - parameter.y_) + parameter.w_;
    }
}
