#ifndef INC_LRENDER_MATRIX_H__
#define INC_LRENDER_MATRIX_H__
/**
@file Matrix.h
@author t-sakai
@date 2016/03/22 create
*/
#include "core.h"

namespace lrender
{
    class Vector3;
    class Vector4;
    class Quaternion;
    class Matrix44;

    //--------------------------------------------
    //---
    //--- Matrix34
    //---
    //--------------------------------------------
    /**
    Left-handed coordinate matrix
    */
    class Matrix34
    {
    public:
        static const Matrix34 Zero;
        static const Matrix34 Identity;

        inline Matrix34();
        Matrix34(
            f32 m00, f32 m01, f32 m02, f32 m03,
            f32 m10, f32 m11, f32 m12, f32 m13,
            f32 m20, f32 m21, f32 m22, f32 m23);

        explicit Matrix34(const Matrix44& rhs);

        void set(f32 m00, f32 m01, f32 m02, f32 m03,
            f32 m10, f32 m11, f32 m12, f32 m13,
            f32 m20, f32 m21, f32 m22, f32 m23);

        inline f32 operator()(s32 r, s32 c) const;

        inline f32& operator()(s32 r, s32 c);

        Matrix34& operator*=(f32 f);
        Matrix34& operator*=(const Matrix34& rhs);
        Matrix34& operator+=(const Matrix34& rhs);
        Matrix34& operator-=(const Matrix34& rhs);

        Matrix34 operator-() const;

        void identity();

        void transpose33();

        f32 determinant33() const;

        void invert33();

        void invert();

        void setTranslate(const Vector3& v);

        void setTranslate(f32 x, f32 y, f32 z);

        void translate(const Vector3& v);

        void translate(const Vector4& v);

        void translate(f32 x, f32 y, f32 z);

        void preTranslate(const Vector3& v);

        void preTranslate(f32 x, f32 y, f32 z);

        void rotateX(f32 radian);

        void rotateY(f32 radian);

        void rotateZ(f32 radian);

        void setRotateAxis(const Vector3& axis, f32 radian);
        void setRotateAxis(f32 x, f32 y, f32 z, f32 radian);

        void setScale(f32 s);
        void setScale(f32 x, f32 y, f32 z);
        void scale(f32 s);
        void scale(f32 x, f32 y, f32 z);

        void mul(const Matrix34& m0, const Matrix34& m1);

        Matrix34& operator*=(const Matrix44& rhs);

        bool isNan() const;

        void getRotation(Quaternion& rotation) const;

        f32 m_[3][4];
    };

    static_assert(std::is_trivially_copyable<Matrix34>::value, "Matrix34 should be trivially copyable");

    inline Matrix34::Matrix34()
    {}

    inline f32 Matrix34::operator()(s32 r, s32 c) const
    {
        LASSERT(0<= r && r < 3);
        LASSERT(0<= c && c < 4);
        return m_[r][c];
    }

    inline f32& Matrix34::operator()(s32 r, s32 c)
    {
        LASSERT(0<= r && r < 3);
        LASSERT(0<= c && c < 4);

        return m_[r][c];
    }

    //--- Matrix34's friend functions
    //--------------------------------------------
    void copy(Matrix34& dst, const Matrix34& src);
    void copy(Matrix34& dst, const Matrix44& src);
    void load(lm128& r0, lm128& r1, lm128& r2, const Matrix34& m);
    void load(lm128& r0, lm128& r1, lm128& r2, const Matrix44& m);
    void store(Matrix34& m, const lm128& r0, const lm128& r1, const lm128& r2);


    //--------------------------------------------
    //---
    //--- Matrix44
    //---
    //--------------------------------------------
    /**
    Left-handed coordinate matrix
    */
    class Matrix44
    {
    public:
        static const Matrix44 Zero;
        static const Matrix44 Idenity;

        inline Matrix44();
        explicit Matrix44(const Matrix34& mat);

        Matrix44(
            f32 m00, f32 m01, f32 m02, f32 m03,
            f32 m10, f32 m11, f32 m12, f32 m13,
            f32 m20, f32 m21, f32 m22, f32 m23,
            f32 m30, f32 m31, f32 m32, f32 m33);

        void set(f32 m00, f32 m01, f32 m02, f32 m03,
            f32 m10, f32 m11, f32 m12, f32 m13,
            f32 m20, f32 m21, f32 m22, f32 m23,
            f32 m30, f32 m31, f32 m32, f32 m33);


        void set(const Vector4& row0, const Vector4& row1, const Vector4& row2, const Vector4& row3);

        inline f32 operator()(s32 r, s32 c) const;
        inline f32& operator()(s32 r, s32 c);

        Matrix44& operator*=(f32 f);
        inline Matrix44& operator*=(const Matrix44& rhs);

        Matrix44& operator+=(const Matrix44& rhs);
        Matrix44& operator-=(const Matrix44& rhs);

        static void copy(Matrix44& dst, const Matrix44& src);
        static void copy(Matrix44& dst, const Matrix34& src);

        inline Matrix44& operator*=(const Matrix34& rhs);

        Matrix44 operator-() const;

        void mul(const Matrix44& m0, const Matrix44& m1);
        void mul(const Matrix34& m0, const Matrix44& m1);
        void mul(const Matrix44& m0, const Matrix34& m1);

        void zero();
        void identity();
        inline void transpose();
        void transpose(const Matrix44& src);
        void getTranspose(Matrix44& dst) const;

        f32 determinant() const;
        inline void invert();
        void getInvert(Matrix44& dst) const;

        void transpose33();

        f32 determinant33() const;

        inline void invert33();
        void getInvert33(Matrix44& dst) const;

        void setTranslate(const Vector3& v);
        void setTranslate(const Vector4& v);

        void setTranslate(f32 x, f32 y, f32 z);

        void translate(const Vector3& v);
        void translate(const Vector4& v);

        void translate(f32 x, f32 y, f32 z);

        void preTranslate(const Vector3& v);

        void preTranslate(f32 x, f32 y, f32 z);

        void rotateX(f32 radian);
        void rotateY(f32 radian);
        void rotateZ(f32 radian);

        void setRotateX(f32 radian);
        void setRotateY(f32 radian);
        void setRotateZ(f32 radian);
        void setRotateAxis(f32 x, f32 y, f32 z, f32 radian);
        void setRotateAxis(const Vector3& axis, f32 radian);

        void setScale(f32 s);
        void scale(f32 s);

        void setScale(f32 x, f32 y, f32 z);
        void scale(f32 x, f32 y, f32 z);

        void lookAt(const Vector4& eye, const Vector4& at, const Vector4& up);
        void lookAt(const Vector3& eye, const Vector3& at, const Vector3& up);

        void lookAt(const Vector4& at);
        void lookAt(const Vector3& at);

        void lookAt(const Vector4& eye, const Quaternion& rotation);
        void lookAt(const Vector3& eye, const Quaternion& rotation);

        void viewPointAlign(const Matrix44& view, const Vector4& position);
        void axisAlign(const Vector4& axis, const Matrix44& view, const Vector4& position);

        /**
        @brief “§Ž‹“Š‰e
        */
        void perspective(f32 width, f32 height, f32 znear, f32 zfar);

        /**
        @brief “§Ž‹“Š‰e
        */
        void perspectiveFov(f32 fovy, f32 aspect, f32 znear, f32 zfar);

        /**
        @brief “§Ž‹“Š‰e. Reverse-Z
        */
        void perspectiveReverseZ(f32 width, f32 height, f32 znear, f32 zfar);

        /**
        @brief “§Ž‹“Š‰e. Reverse-Z
        */
        void perspectiveFovReverseZ(f32 fovy, f32 aspect, f32 znear, f32 zfar);

        /**
        @brief “§Ž‹“Š‰eBƒŠƒjƒA‚y”Å
        */
        void perspectiveLinearZ(f32 width, f32 height, f32 znear, f32 zfar);

        /**
        @brief “§Ž‹“Š‰eBƒŠƒjƒA‚y”Å
        */
        void perspectiveFovLinearZ(f32 fovy, f32 aspect, f32 znear, f32 zfar);

        /**
        @brief •½s“Š‰e
        */
        void ortho(f32 width, f32 height, f32 znear, f32 zfar);
        void orthoOffsetCenter(f32 left, f32 right, f32 top, f32 bottom, f32 znear, f32 zfar);

        void orthoReverseZ(f32 width, f32 height, f32 znear, f32 zfar);
        void orthoOffsetCenterReverseZ(f32 left, f32 right, f32 top, f32 bottom, f32 znear, f32 zfar);

        void getTranslate(f32& x, f32& y, f32& z) const;
        void getTranslate(Vector3& trans) const;
        void getTranslate(Vector4& trans) const;

        void getRow(Vector3& dst, s32 row) const;
        void getRow(Vector4& dst, s32 row) const;

        bool isNan() const;

        void getRotation(Quaternion& rotation) const;

        f32 m_[4][4];

#if defined(LMATH_USE_SSE)
    private:
        //SSEƒZƒbƒgEƒXƒgƒA–½—ß
        inline static void load(lm128& r0, lm128& r1, lm128& r2, lm128& r3, const Matrix44& m);
        inline static void store(Matrix44& m, const lm128& r0, const lm128& r1, const lm128& r2, const lm128& r3);
#endif
    };

    static_assert(std::is_trivially_copyable<Matrix44>::value, "Matrix44 should be trivially copyable");

    inline Matrix44::Matrix44()
    {}

    inline Matrix44& Matrix44::operator*=(const Matrix44& rhs)
    {
        mul(*this, rhs);
        return *this;
    }

    inline f32 Matrix44::operator()(s32 r, s32 c) const
    {
        LASSERT(0 <= r && r < 4);
        LASSERT(0 <= c && c < 4);
        return m_[r][c];
    }

    inline f32& Matrix44::operator()(s32 r, s32 c)
    {
        LASSERT(0 <= r && r < 4);
        LASSERT(0 <= c && c < 4);
        return m_[r][c];
    }

    inline Matrix44& Matrix44::operator*=(const Matrix34& rhs)
    {
        mul(*this, rhs);
        return *this;
    }

    inline void Matrix44::transpose()
    {
        getTranspose(*this);
    }

    inline void Matrix44::invert()
    {
        getInvert(*this);
    }

    inline void Matrix44::invert33()
    {
        getInvert33(*this);
    }

    //--- Matrix44's friend functions
    //------------------------------------------------------------------------------------------------
    void lookAt(Matrix44& view, Matrix44& invview, const Vector4& eye, const Vector4& at, const Vector4& up);
    void lookAt(Matrix44& view, Matrix44& invview, const Vector3& eye, const Vector3& at, const Vector3& up);

    void lookAt(Matrix44& view, Matrix44& invview, const Vector4& at);
    void lookAt(Matrix44& view, Matrix44& invview, const Vector3& at);

    void lookAt(Matrix44& view, Matrix44& invview, const Vector4& eye, const Quaternion& rotation);
    void lookAt(Matrix44& view, Matrix44& invview, const Vector3& eye, const Quaternion& rotation);

    void perspective(Matrix44& proj, Matrix44& invproj, f32 width, f32 height, f32 znear, f32 zfar);
    void perspectiveFov(Matrix44& proj, Matrix44& invproj, f32 fovy, f32 aspect, f32 znear, f32 zfar);

    void perspectiveReverseZ(Matrix44& proj, Matrix44& invproj, f32 width, f32 height, f32 znear, f32 zfar);
    void perspectiveFovReverseZ(Matrix44& proj, Matrix44& invproj, f32 fovy, f32 aspect, f32 znear, f32 zfar);

    void ortho(Matrix44& proj, Matrix44& invproj, f32 width, f32 height, f32 znear, f32 zfar);

    void orthoReverseZ(Matrix44& proj, Matrix44& invproj, f32 width, f32 height, f32 znear, f32 zfar);

    void copy(Matrix44& dst, const Matrix44& src);
    void load(lm128& r0, lm128& r1, lm128& r2, lm128& r3, const Matrix44& m);
    void store(Matrix44& m, const lm128& r0, const lm128& r1, const lm128& r2, const lm128& r3);

    //--------------------------------------------
    //---
    //--- Matrix
    //---
    //--------------------------------------------
    class Matrix
    {
    public:
        Matrix();
        Matrix(s32 r, s32 c);
        ~Matrix();

        inline s32 getRows() const;
        inline s32 getCols() const;
        inline f32* operator[](s32 r);
        inline const f32* operator[](s32 r) const;

        void swap(Matrix& rhs);
        static void mul(Matrix& dst, const Matrix& m0, const Matrix& m1);

        void print();
    private:
       Matrix(const Matrix&) = delete;
       Matrix(Matrix&&) = delete;
       Matrix& operator=(const Matrix&) = delete;
       Matrix& operator=(Matrix&&) = delete;

       s32 rows_;
       s32 cols_;
       f32* data_;
    };

    inline s32 Matrix::getRows() const
    {
        return rows_;
    }

    inline s32 Matrix::getCols() const
    {
        return cols_;
    }

    inline f32* Matrix::operator[](s32 r)
    {
        LASSERT(0<=r && r<rows_);
        return data_ + r*cols_;
    }

    inline const f32* Matrix::operator[](s32 r) const
    {
        LASSERT(0<=r && r<rows_);
        return data_ + r*cols_;
    }
    
    template<class T>
    void randomMatrix(Matrix& m, T& rand)
    {
        for(s32 i=0; i<m.getRows(); ++i){
            for(s32 j=0; j<m.getCols(); ++j){
                m[i][j] = rand.frand2();
            }
        }
    }
}
#endif //INC_LRENDER_MATRIX_H__
