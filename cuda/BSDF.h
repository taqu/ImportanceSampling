#ifndef INC_LCUDA_BSDF_H__
#define INC_LCUDA_BSDF_H__
/**
@file BSDF.h
@author t-sakai
@date 2019/06/24
*/
#include "lcuda.h"
#include "Coordinate.h"

namespace lcuda
{
    class Map
    {
    public:
        LCUDA_DEVICE static float3 toUniformSphere(f32 x, f32 y);
        LCUDA_DEVICE static float3 toUniformHemisphere(f32 x, f32 y);
        LCUDA_DEVICE static float3 toCosineHemisphere(f32 x, f32 y);
        LCUDA_DEVICE static float3 toUniformCone(f32 x, f32 y, f32 cosCutoff);
        LCUDA_DEVICE static float2 toUniformDisk(f32 x, f32 y);
        LCUDA_DEVICE static float2 toUniformTriangle(f32 x, f32 y);
    private:
        Map() = delete;
        ~Map() = delete;
    };

    class PDF
    {
    public:
        LCUDA_DEVICE static inline f32 uniformSphere()
        {
            return INV_PI4;
        }

        LCUDA_DEVICE static inline f32 uniformHemisphere()
        {
            return INV_PI2;
        }

        LCUDA_DEVICE static f32 cosineHemisphere(const float3& v);
        LCUDA_DEVICE static f32 uniformCone(f32 cosCutoff);

        LCUDA_DEVICE static inline f32 uniformDisk()
        {
            return INV_PI;
        }

    private:
        PDF() = delete;
        ~PDF() = delete;
    };

    //-------------------------------------------------------------------
    /**
    */
    class Diffuse
    {
    public:
        LCUDA_DEVICE Diffuse();

        LCUDA_DEVICE float3 evaluate(const float3& wi, const float3& wo) const;

        /**
        @brief Sample isotropic
        */
        LCUDA_DEVICE float3 sample(float3& wo, f32& pdf, const float3& wi, f32 eta0, f32 eta1) const;

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
        LCUDA_DEVICE BSDF(f32 alphax, f32 alphay, f32 extEta, f32 intEta, f32 k);

        LCUDA_DEVICE float3 sample(float3& wo, f32& pdf, const float3& wi, f32 eta0, f32 eta1) const;
    private:
        LCUDA_DEVICE inline float3 reflect(const float3& wi, const float3& n) const
        {
            return 2.0f*dot(wi, n)*n - wi;
        }

        T distribution_;
    };

    template<class T>
    LCUDA_DEVICE BSDF<T>::BSDF(f32 alphax, f32 alphay, f32 extEta, f32 intEta, f32 k)
        :distribution_(alphax, alphay, extEta, intEta, k)
    {}

    template<class T>
    LCUDA_DEVICE float3 BSDF<T>::sample(float3& wo, f32& pdf, const float3& wi, f32 eta0, f32 eta1) const
    {
        float3 m;
        distribution_.sample(m, pdf, wi, eta0, eta1);
        if(pdf<=F32_EPSILON){
            return make_float3(0.0f);
        }

        //f32 f = distribution_.fresnelDielectric(dot(wi, m));
        wo = reflect(wi, m);
        if(LocalCoordinate::cosTheta(wo)<=F32_EPSILON){
            return make_float3(0.0f);
        }

        f32 weight = distribution_.calcWeight(wi, wo, m, pdf);
        pdf /= 4.0f * dot(wo, m);
        return make_float3(weight);
    }

    //-------------------------------------------------------------------
    /**
    Distribution
    */
    class Distribution
    {
    public:
        LCUDA_DEVICE static f32 F(f32 cosTheta, f32 F0);

        LCUDA_DEVICE static f32 fresnelDielectric(f32 cosTheta, f32 eta);

        /**
        @param cosTheta
        @param eta ... 
        @param k ... 
        */
        LCUDA_DEVICE static f32 fresnelConductor(f32 cosTheta, f32 eta, f32 k);
        LCUDA_DEVICE static f32 project2(f32 x, f32 y, const float3& v);

        LCUDA_DEVICE f32 fresnel(f32 cosTheta) const;
        LCUDA_DEVICE f32 fresnelDielectric(f32 cosTheta) const;
        LCUDA_DEVICE f32 fresnelConductor(f32 cosTheta) const;
        LCUDA_DEVICE virtual float3 evaluate(const float3& wi, const float3& wo) const =0;
        LCUDA_DEVICE virtual void sample(float3& m, f32& pdf, const float3& wi, f32 eta0, f32 eta1) const =0;
        LCUDA_DEVICE virtual f32 calcWeight(const float3& wi, const float3& wo, const float3& m, f32 pdf) const =0;
    protected:
        LCUDA_DEVICE Distribution(f32 alphax, f32 alphay, f32 extEta, f32 intEta, f32 k);
        LCUDA_DEVICE virtual ~Distribution();

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
        LCUDA_DEVICE BeckmannIsotropic(f32 alphax, f32 alphay, f32 extEta, f32 intEta, f32 k);

        LCUDA_DEVICE virtual float3 evaluate(const float3& wi, const float3& wo) const override;

        /**
        @brief Sample a next direction
        */
        LCUDA_DEVICE virtual void sample(float3& m, f32& pdf, const float3& wi, f32 eta0, f32 eta1) const override;

        LCUDA_DEVICE virtual f32 calcWeight(const float3& wi, const float3& wo, const float3& m, f32 pdf) const override;

    private:
        LCUDA_DEVICE f32 G1(const float3& v, const float3& m) const;

        LCUDA_DEVICE f32 D(const float3& m, const float3& n) const;
        LCUDA_DEVICE f32 D(const float3& m) const;
    };

    //-------------------------------------------------------------------
    /**
    Isotropic GGX distribution
    */
    class GGXIsotropic : public Distribution
    {
    public:
        LCUDA_DEVICE GGXIsotropic(f32 alphax, f32 alphay, f32 extEta, f32 intEta, f32 k);

        LCUDA_DEVICE virtual float3 evaluate(const float3& wi, const float3& wo) const override;

        /**
        @brief Sample a next direction
        */
        LCUDA_DEVICE virtual void sample(float3& m, f32& pdf, const float3& wi, f32 eta0, f32 eta1) const override;

        LCUDA_DEVICE virtual f32 calcWeight(const float3& wi, const float3& wo, const float3& m, f32 pdf) const override;

    private:
        LCUDA_DEVICE f32 D(const float3& m, const float3& n) const;

        LCUDA_DEVICE f32 G1(const float3& v, const float3& m) const;
        LCUDA_DEVICE f32 D(const float3& m) const;
    };

    //-------------------------------------------------------------------
    /**
    Sampling anisotropic GGX distribution with using the distribution of visible normals
    */
    class GGXAnisotropicVND : public Distribution
    {
    public:
        LCUDA_DEVICE GGXAnisotropicVND(f32 alphax, f32 alphay, f32 extEta, f32 intEta, f32 k);

        LCUDA_DEVICE virtual float3 evaluate(const float3& wi, const float3& wo) const override;

        /**
        @brief Sample a view dependent normal
        @param 
        */
        LCUDA_DEVICE virtual void sample(float3& m, f32& pdf, const float3& wi, f32 eta0, f32 eta1) const override;

        LCUDA_DEVICE virtual f32 calcWeight(const float3& wi, const float3& wo, const float3& m, f32 pdf) const override;

    protected:
        LCUDA_DEVICE f32 G1(const float3& v, const float3& m) const;

        LCUDA_DEVICE f32 D(const float3& m, const float3& n) const;
        LCUDA_DEVICE f32 D(const float3& m) const;

        /**
        @brief Inverse slope CDF
        */
        LCUDA_DEVICE void sample11(f32& slopex, f32& slopey, f32 thetai, f32 eta0, f32 eta1) const;
    };

    //-------------------------------------------------------------------
    /**
    Sampling on anisotropic GGX VNDF with using elipsoid
    */
    class GGXAnisotropicElipsoidVND : public GGXAnisotropicVND
    {
    public:
        LCUDA_DEVICE GGXAnisotropicElipsoidVND(f32 alphax, f32 alphay, f32 extEta, f32 intEta, f32 k);

        LCUDA_DEVICE virtual float3 evaluate(const float3& wi, const float3& wo) const override;

        /**
        @brief Sample a view dependent normal
        @param 
        */
        LCUDA_DEVICE virtual void sample(float3& m, f32& pdf, const float3& wi, f32 eta0, f32 eta1) const override;
    };
}
#endif //INC_LCUDA_BSDF_H__
