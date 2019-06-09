#ifndef INC_LRENDER_VECTOR_H_
#define INC_LRENDER_VECTOR_H_
/**
@file Vector.h
@author t-sakai
@date 2019/03/03 create
*/
#include "core.h"

namespace lrender
{
    class Vector4;
    class Quaternion;
    class Matrix34;
    class Matrix44;

    //--------------------------------------------
    //---
    //--- Vector2
    //---
    //--------------------------------------------
    class Vector2
    {
    public:
        static const Vector2 Zero;
        static const Vector2 One;

        inline Vector2();

        explicit inline Vector2(f32 xy);

        inline Vector2(f32 x, f32 y);

        inline void zero();
        inline void one();
        inline void set(f32 x, f32 y);

        inline f32 operator[](s32 index) const;
        inline f32& operator[](s32 index);
        inline Vector2 operator-() const;

        Vector2& operator+=(const Vector2& v);
        Vector2& operator-=(const Vector2& v);

        Vector2& operator*=(f32 f);
        Vector2& operator/=(f32 f);

        bool isEqual(const Vector2& v) const;
        bool isEqual(const Vector2& v, f32 epsilon) const;

        inline bool operator==(const Vector2& v) const;
        inline bool operator!=(const Vector2& v) const;

        inline f32 length() const;
        f32 lengthSqr() const;

        bool isNan() const;

        f32 x_, y_;
    };

    static_assert(std::is_trivially_copyable<Vector2>::value, "Vector2 should be trivially copyable");

    inline Vector2::Vector2()
    {}

    inline Vector2::Vector2(f32 xy)
        :x_(xy)
        ,y_(xy)
    {}

    inline Vector2::Vector2(f32 x, f32 y)
        :x_(x)
        ,y_(y)
    {}

    inline void Vector2::zero()
    {
        x_ = y_ = 0.0f;
    }

    inline void Vector2::one()
    {
        x_ = y_ = 1.0f;
    }

    inline void Vector2::set(f32 x, f32 y)
    {
        x_ = x;
        y_ = y;
    }

    inline f32 Vector2::operator[](s32 index) const
    {
        LASSERT(0<=index && index < 2);
        return (&x_)[index];
    }

    inline f32& Vector2::operator[](s32 index)
    {
        LASSERT(0<=index && index < 2);
        return (&x_)[index];
    }

    inline Vector2 Vector2::operator-() const
    {
        return {-x_, -y_};
    }

    inline bool Vector2::operator==(const Vector2& v) const
    {
        return isEqual(v);
    }

    inline bool Vector2::operator!=(const Vector2& v) const
    {
        return !isEqual(v);
    }

    inline f32 Vector2::length() const
    {
        return ::sqrtf(lengthSqr());
    }

    //--- Vector2's friend functions
    //--------------------------------------------------
    Vector2 normalize(const Vector2& v);

    Vector2 normalizeChecked(const Vector2& v);

    Vector2 operator+(const Vector2& v0, const Vector2& v1);

    Vector2 operator-(const Vector2& v0, const Vector2& v1);

    Vector2 operator*(f32 f, const Vector2& v);

    Vector2 operator*(const Vector2& v, f32 f);

    Vector2 operator/(const Vector2& v, f32 f);

    f32 dot(const Vector2& v0, const Vector2& v1);

    f32 distanceSqr(const Vector2& v0, const Vector2& v1);

    f32 distance(const Vector2& v0, const Vector2& v1);

    Vector2 lerp(const Vector2& v0, const Vector2& v1, f32 t);


    Vector2 add(const Vector2& v0, const Vector2& v1);

    Vector2 sub(const Vector2& v0, const Vector2& v1);

    Vector2 mul(f32 f, const Vector2& v);

    Vector2 mul(const Vector2& v, f32 f);

    Vector2 muladd(f32 f, const Vector2& v0, const Vector2& v1);

    Vector2 minimum(const Vector2& v0, const Vector2& v1);

    Vector2 maximum(const Vector2& v0, const Vector2& v1);

    f32 minimum(const Vector2& v);

    f32 maximum(const Vector2& v);


    //--------------------------------------------
    //---
    //--- Vector3
    //---
    //--------------------------------------------
    class Vector3
    {
    public:
        static const Vector3 Zero;
        static const Vector3 One;
        static const Vector3 Forward;
        static const Vector3 Backward;
        static const Vector3 Up;
        static const Vector3 Down;
        static const Vector3 Right;
        static const Vector3 Left;

        inline Vector3();
        explicit Vector3(f32 xyz);
        Vector3(f32 x, f32 y, f32 z);
        explicit Vector3(const Vector4& v);
        explicit Vector3(const lm128& v);

        void zero();
        void one();

        void set(f32 x, f32 y, f32 z);

        void set(const Vector4& v);

        inline f32 operator[](s32 index) const;
        inline f32& operator[](s32 index);
        inline Vector3 operator-() const;

        Vector3& operator+=(const Vector3& v);
        Vector3& operator-=(const Vector3& v);

        Vector3& operator*=(f32 f);
        Vector3& operator/=(f32 f);

        Vector3& operator*=(const Vector3& v);
        Vector3& operator/=(const Vector3& v);

        bool isEqual(const Vector3& v) const;
        bool isEqual(const Vector3& v, f32 epsilon) const;

        inline bool operator==(const Vector3& v) const;
        inline bool operator!=(const Vector3& v) const;

        inline f32 length() const;
        f32 lengthSqr() const;

        void swap(Vector3& rhs);

        bool isNan() const;

        f32 x_;
        f32 y_;
        f32 z_;
    };

    static_assert(std::is_trivially_copyable<Vector3>::value, "Vector3 should be trivially copyable");

    inline Vector3::Vector3()
    {}

    inline f32 Vector3::operator[](s32 index) const
    {
        LASSERT(0<=index && index < 3);
        return (&x_)[index];
    }

    inline f32& Vector3::operator[](s32 index)
    {
        LASSERT(0<=index && index < 3);
        return (&x_)[index];
    }

    inline Vector3 Vector3::operator-() const
    {
        return {-x_, -y_, -z_};
    }

    inline bool Vector3::operator==(const Vector3& v) const
    {
        return isEqual(v);
    }

    inline bool Vector3::operator!=(const Vector3& v) const
    {
        return !isEqual(v);
    }

    inline f32 Vector3::length() const
    {
        return ::sqrtf(lengthSqr());
    }

    //--- Vector3's friend functions
    //--------------------------------------------------
    inline lm128 load(const Vector3& v)
    {
        return load3(&v.x_);
    }

    inline lm128 load(f32 x, f32 y, f32 z)
    {
        return set_m128(x, y, z, 1.0f);
    }

    inline lm128 load(f32 v)
    {
        return _mm_set1_ps(v);
    }

    inline void store(Vector3& v, const lm128& r)
    {
        store3(&v.x_, r);
    }

    Vector3 operator+(const Vector3& v0, const Vector3& v1);
    Vector3 operator-(const Vector3& v0, const Vector3& v1);
    Vector3 operator*(f32 f, const Vector3& v);
    Vector3 operator*(const Vector3& v, f32 f);
    Vector3 operator*(const Vector3& v0, const Vector3& v1);
    Vector3 operator/(const Vector3& v, f32 f);
    Vector3 operator/(const Vector3& v0, const Vector3& v1);

    Vector3 normalize(const Vector3& v);
    Vector3 normalize(const Vector3& v, f32 lengthSqr);
    Vector3 normalizeChecked(const Vector3& v);
    Vector3 normalizeChecked(const Vector3& v, const Vector3& default);

    Vector3 absolute(const Vector3& v);

    f32 dot(const Vector3& v0, const Vector3& v1);

    f32 distanceSqr(const Vector3& v0, const Vector3& v1);
    inline f32 distance(const Vector3& v0, const Vector3& v1)
    {
        return ::sqrtf(distanceSqr(v0, v1));
    }

    Vector3 cross(const Vector3& v0, const Vector3& v1);

    /**
    @brief Linear interpolation v = (1-t)*v0 + t*v1
    @param v0 ...
    @param v1 ...
    */
    Vector3 lerp(const Vector3& v0, const Vector3& v1, f32 t);

    /**
    @brief Linear interpolation v = t1*v0 + t0*v1
    @param v0 ...
    @param v1 ...
    */
    Vector3 lerp(const Vector3& v0, const Vector3& v1, f32 t0, f32 t1);

    inline Vector3 mul(f32 f, const Vector3& v)
    {
        return f*v;
    }

    inline Vector3 mul(const Vector3& v, f32 f)
    {
        return v*f;
    }

    Vector3 mul(const Matrix34& m, const Vector3& v);
    Vector3 mul(const Vector3& v, const Matrix34& m);

    Vector3 mul33(const Matrix34& m, const Vector3& v);
    Vector3 mul33(const Vector3& v, const Matrix34& m);

    Vector3 mul33(const Matrix44& m, const Vector3& v);
    Vector3 mul33(const Vector3& v, const Matrix44& m);

    Vector3 rotate(const Vector3& v, const Quaternion& rotation);
    Vector3 rotate(const Quaternion& rotation, const Vector3& v);


    inline Vector3 add(const Vector3& v0, const Vector3& v1)
    {
        return v0+v1;
    }

    inline Vector3 sub(const Vector3& v0, const Vector3& v1)
    {
        return v0-v1;
    }

    Vector3 mul(const Vector3& v0, const Vector3& v1);
    Vector3 div(const Vector3& v0, const Vector3& v1);


    Vector3 minimum(const Vector3& v0, const Vector3& v1);

    Vector3 maximum(const Vector3& v0, const Vector3& v1);

    f32 minimum(const Vector3& v);

    f32 maximum(const Vector3& v);

    // v0*v1 + v2
    Vector3 muladd(const Vector3& v0, const Vector3& v1, const Vector3& v2);

    // a*v1 + v2
    Vector3 muladd(f32 a, const Vector3& v0, const Vector3& v1);


    //--------------------------------------------
    //---
    //--- Vector4
    //---
    //--------------------------------------------
    class Vector4
    {
    public:
        static const Vector4 Zero;
        static const Vector4 One;
        static const Vector4 Identity;
        static const Vector4 Forward;
        static const Vector4 Backward;
        static const Vector4 Up;
        static const Vector4 Down;
        static const Vector4 Right;
        static const Vector4 Left;

        Vector4(){}
        explicit Vector4(f32 xyzw);
        Vector4(f32 x, f32 y, f32 z);
        Vector4(f32 x, f32 y, f32 z, f32 w);
        explicit Vector4(const Vector3& v);
        Vector4(const Vector3& v, f32 w);
        explicit inline Vector4(const lm128& v);

        void zero();
        void one();
        void identity();

        void set(f32 x, f32 y, f32 z, f32 w);
        void set(const Vector3& v);
        void set(const Vector3& v, f32 w);
        void set(f32 v);
        void set(const lm128& v);

        inline f32 operator[](s32 index) const;
        inline f32& operator[](s32 index);

        Vector4 operator-() const;

        Vector4& operator+=(const Vector4& v);
        Vector4& operator-=(const Vector4& v);

        Vector4& operator*=(f32 f);
        Vector4& operator/=(f32 f);

        Vector4& operator*=(const Vector4& v);
        Vector4& operator/=(const Vector4& v);

        bool isEqual(const Vector4& v) const;
        bool isEqual(const Vector4& v, f32 epsilon) const;

        inline bool operator==(const Vector4& v) const;
        inline bool operator!=(const Vector4& v) const;

        void setLength();
        f32 length() const;
        f32 lengthSqr() const;

        void swap(Vector4& rhs);

        bool isNan() const;
        bool isZero() const;

        f32 x_, y_, z_, w_;
    };

    //--------------------------------------------
    //---
    //--- Vector4
    //---
    //--------------------------------------------
    static_assert(std::is_trivially_copyable<Vector4>::value, "Vector4 should be trivially copyable");

    inline Vector4::Vector4(const lm128& v)
    {
        _mm_storeu_ps(&x_, v);
    }

    inline f32 Vector4::operator[](s32 index) const
    {
        LASSERT(0<=index && index < 4);
        return (&x_)[index];
    }

    inline f32& Vector4::operator[](s32 index)
    {
        LASSERT(0<=index && index < 4);
        return (&x_)[index];
    }

    inline bool Vector4::operator==(const Vector4& v) const
    {
        return isEqual(v);
    }

    inline bool Vector4::operator!=(const Vector4& v) const
    {
        return !isEqual(v);
    }

    //--- Vector4's friend functions
    //--------------------------------------------------
    inline lm128 load(const Vector4& v)
    {
        return _mm_loadu_ps(&v.x_);
    }

    inline void store(Vector4& v, const lm128& r)
    {
        _mm_storeu_ps(&v.x_, r);
    }

    void copy(Vector4& dst, const Vector4& src);

    Vector4 normalize(const Vector4& v);
    Vector4 normalize(const Vector4& v, f32 lengthSqr);
    Vector4 normalizeChecked(const Vector4& v);
    Vector4 absolute(const Vector4& v);

    f32 dot(const Vector4& v0, const Vector4& v1);
    Vector4 cross3(const Vector4& v0, const Vector4& v1);
    f32 distanceSqr(const Vector4& v0, const Vector4& v1);
    f32 distance(const Vector4& v0, const Vector4& v1);
    f32 manhattanDistance(const Vector4& v0, const Vector4& v1);

    f32 distanceSqr3(const Vector3& v0, const Vector4& v1);
    inline f32 distanceSqr3(const Vector4& v0, const Vector3& v1)
    {
        return distanceSqr3(v1, v0);
    }
    f32 distanceSqr3(const Vector4& v0, const Vector4& v1);
    f32 distance3(const Vector3& v0, const Vector4& v1);
    inline f32 distance3(const Vector4& v0, const Vector3& v1)
    {
        return distance3(v1, v0);
    }
    f32 distance3(const Vector4& v0, const Vector4& v1);
    f32 manhattanDistance3(const Vector3& v0, const Vector4& v1);
    inline f32 manhattanDistance3(const Vector4& v0, const Vector3& v1)
    {
        return manhattanDistance3(v1, v0);
    }
    f32 manhattanDistance3(const Vector4& v0, const Vector4& v1);

    Vector4 mul(const Matrix34& m, const Vector4& v);
    Vector4 mul(const Vector4& v, const Matrix34& m);

    Vector4 mul(const Matrix44& m, const Vector4& v);
    Vector4 mul(const Vector4& v, const Matrix44& m);

    lm128 mul(const lm128& m0, const lm128& m1, const lm128& m2, const lm128& m3,
        const lm128& tv0, const lm128& tv1, const lm128& tv2, const lm128& tv3);
    lm128 mul(const lm128& m0, const lm128& m1, const lm128& m2, const lm128& m3,
        const lm128& tv);

    Vector4 mulPoint(const Matrix44& m, const Vector4& v);
    Vector4 mulVector(const Matrix44& m, const Vector4& v);

    Vector4 rotate(const Vector4& v, const Quaternion& rotation);
    Vector4 rotate(const Quaternion& rotation, const Vector4& v);

    Vector4 operator+(const Vector4& v0, const Vector4& v1);
    Vector4 operator-(const Vector4& v0, const Vector4& v1);
    Vector4 operator*(f32 f, const Vector4& v);

    inline Vector4 operator*(const Vector4& v, f32 f)
    {
        return f*v;
    }

    Vector4 operator*(const Vector4& v0, const Vector4& v1);

    Vector4 operator/(const Vector4& v, f32 f);
    Vector4 operator/(const Vector4& v0, const Vector4& v1);

    inline Vector4 add(const Vector4& v0, const Vector4& v1)
    {
        return v0+v1;
    }

    inline Vector4 sub(const Vector4& v0, const Vector4& v1)
    {
        return v0-v1;
    }

    Vector4 mul(const Vector4& v0, const Vector4& v1);

    inline Vector4 mul(f32 f, const Vector4& v)
    {
        return f*v;
    }

    inline Vector4 mul(const Vector4& v, f32 f)
    {
        return v*f;
    }

    Vector4 div(const Vector4& v0, const Vector4& v1);

    inline Vector4 div(const Vector4& v0, f32 f)
    {
        return v0/f;
    }

    Vector4 add(const Vector4& v, f32 f);
    Vector4 sub(const Vector4& v, f32 f);

    Vector4 minimum(const Vector4& v0, const Vector4& v1);
    Vector4 maximum(const Vector4& v0, const Vector4& v1);

    f32 minimum(const Vector4& v);
    f32 maximum(const Vector4& v);

    /**
    @brief v0*v1 + v2
    */
    Vector4 muladd(const Vector4& v0, const Vector4& v1, const Vector4& v2);

    /**
    @brief a*v0 + v1
    */
    Vector4 muladd(f32 a, const Vector4& v0, const Vector4& v1);

    Vector4 floor(const Vector4& v);
    Vector4 ceil(const Vector4& v);

    Vector4 invert(const Vector4& v);
    Vector4 sqrt(const Vector4& v);

    /**
    @brief v0 * (1-t) + v1 * t
    */
    Vector4 lerp(const Vector4& v0, const Vector4& v1, f32 t);

    /**
    @brief v0 * (1-t) + v1 * t
    */
    Vector4 slerp(const Vector4& v0, const Vector4& v1, f32 t);

    /**
    @brief v0 * (1-t) + v1 * t
    */
    Vector4 slerp(const Vector4& v0, const Vector4& v1, f32 t, f32 cosine);

    Vector4 getParallelComponent(const Vector4& v, const Vector4& basis);

    Vector4 getPerpendicularComponent(const Vector4& v, const Vector4& basis);

    Vector4 getLinearZParameterReverseZ(f32 znear, f32 zfar);

    f32 toLinearZ(f32 z, const Vector4& parameter);
}
#endif //INC_LRENDER_VECTOR_H_
