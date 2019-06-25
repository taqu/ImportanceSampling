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

    class Map
    {
    public:
        static Vector3 toUniformSphere(f32 x, f32 y);
        static Vector3 toUniformHemisphere(f32 x, f32 y);
        static Vector3 toCosineHemisphere(f32 x, f32 y);
        static Vector3 toUniformCone(f32 x, f32 y, f32 cosCutoff);
        static Vector2 toUniformDisk(f32 x, f32 y);
        static Vector2 toUniformTriangle(f32 x, f32 y);
    private:
        Map() = delete;
        ~Map() = delete;
    };

    class PDF
    {
    public:
        static inline f32 uniformSphere()
        {
            return INV_PI4;
        }

        static inline f32 uniformHemisphere()
        {
            return INV_PI2;
        }

        static f32 cosineHemisphere(const Vector3& v);
        static f32 uniformCone(f32 cosCutoff);

        static inline f32 uniformDisk()
        {
            return INV_PI;
        }

    private:
        PDF() = delete;
        ~PDF() = delete;
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
    */
    class Diffuse
    {
    public:
        Diffuse();

        Vector3 evaluate(const Vector3& wi, const Vector3& wo) const;

        /**
        @brief Sample isotropic
        */
        Vector3 sample(Vector3& wo, f32& pdf, const Vector3& wi, f32 eta0, f32 eta1) const;

    protected:
    };

    //-------------------------------------------------------------------
    /**
    BSDF
    */
    template<class T>
    class BSDF
    {
    public:
        BSDF(f32 alphax, f32 alphay, f32 extEta, f32 intEta, f32 k);

        Vector3 sample(Vector3& wo, f32& pdf, const Vector3& wi, f32 eta0, f32 eta1) const;
    private:
        inline Vector3 reflect(const Vector3& wi, const Vector3& n) const
        {
            return 2.0f*dot(wi, n)*n - wi;
        }

        T distribution_;
    };

    template<class T>
    BSDF<T>::BSDF(f32 alphax, f32 alphay, f32 extEta, f32 intEta, f32 k)
        :distribution_(alphax, alphay, extEta, intEta, k)
    {}

    template<class T>
    Vector3 BSDF<T>::sample(Vector3& wo, f32& pdf, const Vector3& wi, f32 eta0, f32 eta1) const
    {
        Vector3 m;
        distribution_.sample(m, pdf, wi, eta0, eta1);
        if(pdf<=F32_EPSILON){
            return Vector3::Zero;
        }

        //f32 f = distribution_.fresnelDielectric(dot(wi, m));
        wo = reflect(wi, m);
        if(LocalCoordinate::cosTheta(wo)<=F32_EPSILON){
            return Vector3::Zero;
        }

        f32 weight = distribution_.calcWeight(wi, wo, m, pdf);
        pdf /= 4.0f * dot(wo, m);
        return Vector3(weight);
    }

    //-------------------------------------------------------------------
    /**
    Distribution
    */
    class Distribution
    {
    public:
        static const Vector3 Normal;

        static f32 F(f32 cosTheta, f32 F0);

        static f32 fresnelDielectric(f32 cosTheta, f32 eta);

        /**
        @param cosTheta
        @param eta ... 
        @param k ... 
        */
        static f32 fresnelConductor(f32 cosTheta, f32 eta, f32 k);
        static f32 project2(f32 x, f32 y, const Vector3& v);

        f32 fresnel(f32 cosTheta) const;
        f32 fresnelDielectric(f32 cosTheta) const;
        f32 fresnelConductor(f32 cosTheta) const;
        virtual Vector3 evaluate(const Vector3& wi, const Vector3& wo) const =0;
        virtual void sample(Vector3& m, f32& pdf, const Vector3& wi, f32 eta0, f32 eta1) const =0;
        virtual f32 calcWeight(const Vector3& wi, const Vector3& wo, const Vector3& m, f32 pdf) const =0;
    protected:
        Distribution(f32 alphax, f32 alphay, f32 extEta, f32 intEta, f32 k);
        virtual ~Distribution();

        f32 alphax_;
        f32 alphay_;
        f32 extEta_;
        f32 intEta_;
        f32 eta_; //(int refraction index)/(ext refraction index)
        f32 k_; //(absorption coefficient)/(ext refraction index)
        f32 F0_;
    };

    //-------------------------------------------------------------------
    /**
    Isotropic Beckmann distribution
    */
    class BeckmannIsotropic : public Distribution
    {
    public:
        BeckmannIsotropic(f32 alphax, f32 alphay, f32 extEta, f32 intEta, f32 k);

        virtual Vector3 evaluate(const Vector3& wi, const Vector3& wo) const override;

        /**
        @brief Sample a next direction
        */
        virtual void sample(Vector3& m, f32& pdf, const Vector3& wi, f32 eta0, f32 eta1) const override;

        virtual f32 calcWeight(const Vector3& wi, const Vector3& wo, const Vector3& m, f32 pdf) const override;

    private:
        f32 G1(const Vector3& v, const Vector3& m) const;

        f32 D(const Vector3& m, const Vector3& n) const;
        f32 D(const Vector3& m) const;
    };

    //-------------------------------------------------------------------
    /**
    Isotropic GGX distribution
    */
    class GGXIsotropic : public Distribution
    {
    public:
        GGXIsotropic(f32 alphax, f32 alphay, f32 extEta, f32 intEta, f32 k);

        virtual Vector3 evaluate(const Vector3& wi, const Vector3& wo) const override;

        /**
        @brief Sample a next direction
        */
        virtual void sample(Vector3& m, f32& pdf, const Vector3& wi, f32 eta0, f32 eta1) const override;

        virtual f32 calcWeight(const Vector3& wi, const Vector3& wo, const Vector3& m, f32 pdf) const override;

    private:
        f32 D(const Vector3& m, const Vector3& n) const;

        f32 G1(const Vector3& v, const Vector3& m) const;
        f32 D(const Vector3& m) const;
    };

    //-------------------------------------------------------------------
    /**
    Sampling anisotropic GGX distribution with using the distribution of visible normals
    */
    class GGXAnisotropicVND : public Distribution
    {
    public:
        GGXAnisotropicVND(f32 alphax, f32 alphay, f32 extEta, f32 intEta, f32 k);

        virtual Vector3 evaluate(const Vector3& wi, const Vector3& wo) const override;

        /**
        @brief Sample a view dependent normal
        @param 
        */
        virtual void sample(Vector3& m, f32& pdf, const Vector3& wi, f32 eta0, f32 eta1) const override;

        virtual f32 calcWeight(const Vector3& wi, const Vector3& wo, const Vector3& m, f32 pdf) const override;

    protected:
        f32 G1(const Vector3& v, const Vector3& m) const;

        f32 D(const Vector3& m, const Vector3& n) const;
        f32 D(const Vector3& m) const;

        /**
        @brief Inverse slope CDF
        */
        void sample11(f32& slopex, f32& slopey, f32 thetai, f32 eta0, f32 eta1) const;
    };

    //-------------------------------------------------------------------
    /**
    Sampling on anisotropic GGX VNDF with using elipsoid
    */
    class GGXAnisotropicElipsoidVND : public GGXAnisotropicVND
    {
    public:
        GGXAnisotropicElipsoidVND(f32 alphax, f32 alphay, f32 extEta, f32 intEta, f32 k);

        virtual Vector3 evaluate(const Vector3& wi, const Vector3& wo) const override;

        /**
        @brief Sample a view dependent normal
        @param 
        */
        virtual void sample(Vector3& m, f32& pdf, const Vector3& wi, f32 eta0, f32 eta1) const override;
    };
}
