#include "core.h"
#include "Vector.h"

namespace lrender
{
    class Ray
    {
    public:
        Ray(){}
        Ray(const Vector3& origin, const Vector3& direction)
            :origin_(origin)
            ,direction_(direction)
        {}

        Vector3 origin_;
        Vector3 direction_;
    };

    //---------------------------------------------
    //---
    //--- Xoshiro128+
    //---
    //---------------------------------------------
    class Xoshiro128Plus
    {
    public:
        Xoshiro128Plus();
        explicit Xoshiro128Plus(u32 seed);
        ~Xoshiro128Plus();

        /**
        @brief ã[éóóêêîê∂ê¨äÌèâä˙âª
        @param seed
        */
        void srand(u32 seed);

        /**
        @brief 0 - 0xFFFFFFFFUÇÃóêêîê∂ê¨
        */
        u32 rand();

        /**
        @brief (0, 1]ÇÃóêêîê∂ê¨
        */
        f32 frand();

        /**
        @brief [0, 1)ÇÃóêêîê∂ê¨
        */
        f32 frand2();
    private:
        static const u32 N = 4;
        u32 state_[N];
    };

    //-------------------------------------------------------------------
    /**
    BSDF
    */
    class BSDF
    {
    public:
        static const Vector3 Normal;

        static f32 F(const Vector3& wi, const Vector3& m, f32 F0);

        virtual Vector3 evaluate(const Vector3& wi, const Vector3& wo) const =0;
        virtual void sample(Vector3& wm, f32 eta0, f32 eta1) const =0;
        virtual f32 calcWeight(const Vector3& wi, const Vector3& wo) const;
    protected:
        BSDF()
        {}
        virtual ~BSDF()
        {}

        virtual f32 G1(const Vector3& v) const =0;
    };

    //-------------------------------------------------------------------
    /**
    */
    class Diffuse : public BSDF
    {
    public:
        Diffuse();

        virtual Vector3 evaluate(const Vector3& wi, const Vector3& wo) const override;

        /**
        @brief Sample isotropic
        */
        virtual void sample(Vector3& m, f32 eta0, f32 eta1) const override;

        virtual f32 calcWeight(const Vector3& wi, const Vector3& wo) const override;

    protected:
        virtual f32 G1(const Vector3& v) const override
        {
            return 0.0f;
        }
    };

    //-------------------------------------------------------------------
    /**
    Isotropic Beckmann distribution
    */
    class BeckmannIsotropic : public BSDF
    {
    public:
        BeckmannIsotropic(f32 alpha, f32 eta0, f32 eta1);

        virtual Vector3 evaluate(const Vector3& wi, const Vector3& wo) const override;

        /**
        @brief Sample a next direction
        */
        virtual void sample(Vector3& wm, f32 eta0, f32 eta1) const override;

        f32 D(const Vector3& m, const Vector3& n) const;

    protected:
        virtual f32 G1(const Vector3& v) const override;
    private:
        f32 alpha_;
        f32 eta0_;
        f32 eta1_;
        f32 F0_;
    };

    //-------------------------------------------------------------------
    /**
    Isotropic GGX distribution
    */
    class GGXIsotropic : public BSDF
    {
    public:
        GGXIsotropic(f32 alpha, f32 eta0, f32 eta1);

        virtual Vector3 evaluate(const Vector3& wi, const Vector3& wo) const override;

        /**
        @brief Sample a next direction
        */
        virtual void sample(Vector3& wm, f32 eta0, f32 eta1) const override;

        f32 D(const Vector3& m, const Vector3& n) const;

    protected:
        virtual f32 G1(const Vector3& v) const override;
    private:
        f32 alpha_;
        f32 eta0_;
        f32 eta1_;
        f32 F0_;
    };
}
