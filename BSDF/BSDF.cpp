#include "BSDF.h"
#include <limits>
#include "Coordinate.h"

namespace lrender
{
    namespace
    {
        inline u32 rotl(u32 x, s32 k)
        {
            return (x << k) | (x >> (32 - k));
        }

        // Return (0, 1]
        inline f32 toF32_0(u32 x)
        {
            static const u32 m0 = 0x3F800000U;
            static const u32 m1 = 0x007FFFFFU;
            x = m0|(x&m1);
            return (*(f32*)&x)- 0.999999881f;
        }

        // Return [0, 1)
        inline f32 toF32_1(u32 x)
        {
            static const u32 m0 = 0x3F800000U;
            static const u32 m1 = 0x007FFFFFU;
            x = m0|(x&m1);
            return (*(f32*)&x) - 1.000000000f;
        }
    }

    //---------------------------------------------
    //---
    //--- Xoshiro128Plus
    //---
    //---------------------------------------------
    Xoshiro128Plus::Xoshiro128Plus()
        :state_{123456789, 362436069, 521288629, 88675123}
    {}

    Xoshiro128Plus::Xoshiro128Plus(u32 seed)
    {
        srand(seed);
    }
    Xoshiro128Plus::~Xoshiro128Plus()
    {}

    void Xoshiro128Plus::srand(u32 seed)
    {
        state_[0] = seed;
        for(u32 i=1; i<N; ++i){
            state_[i] = (1812433253 * (state_[i-1]^(state_[i-1] >> 30)) + i);
        }
    }

    u32 Xoshiro128Plus::rand()
    {
        const u32 result = state_[0] + state_[3];
        const u32 t = state_[1] << 9;

        state_[2] ^= state_[0];
        state_[3] ^= state_[1];
        state_[1] ^= state_[2];
        state_[0] ^= state_[3];

        state_[2] ^= t;

        state_[3] = rotl(state_[3], 11);

        return result;
    }

    f32 Xoshiro128Plus::frand()
    {
        return toF32_0(rand());
    }

    f32 Xoshiro128Plus::frand2()
    {
        return toF32_1(rand());
    }

    //-------------------------------------------------------------------
    //---
    //--- BSDF
    //---
    //-------------------------------------------------------------------
    f32 BSDF::F(const Vector3& wi, const Vector3& m, f32 F0)
    {
        LASSERT(isEqual(wi.length(), 1.0f, F32_EPSILON*2));
        LASSERT(isEqual(m.length(), 1.0f, F32_EPSILON*2));
        LASSERT(0.0f<=F0);
        f32 r0 = F0*F0;
        f32 cosTheta = dot(wi,m);
        f32 c = (1-cosTheta);
        f32 c2 = c*c;
        f32 c5 = c2*c2*c;
        return r0 + (1.0f-r0)*c5;
    }

    f32 BSDF::calcWeight(const Vector3& wi, const Vector3& wo) const
    {
        LASSERT(isEqual(wi.length(), 1.0f));
        LASSERT(isEqual(wo.length(), 1.0f));

        const Vector3 hr = normalizeChecked(wi + wo, Normal);

        f32 g = G1(wi) * G1(wo);

        f32 cosWiWm = dot(wi, hr);
        f32 cosWi = LocalCoordinate::cosTheta(wi);
        f32 cosWm = LocalCoordinate::cosTheta(hr);
        return absolute(cosWiWm * g) / maximum(absolute(cosWi * cosWm), 1.0e-7f);
    }

    //-------------------------------------------------------------------
    //---
    //--- BSDF
    //---
    //-------------------------------------------------------------------
    const Vector3 BSDF::Normal  = Vector3(0.0f, 0.0f, 1.0f);

    //-------------------------------------------------------------------
    //---
    //--- Diffuse
    //---
    //-------------------------------------------------------------------
    Diffuse::Diffuse()
    {}

    Vector3 Diffuse::evaluate(const Vector3& wi, const Vector3& wo) const
    {
        return Vector3(INV_PI);
    }

    void Diffuse::sample(Vector3& wm, f32 eta0, f32 eta1) const
    {
        f32 phi = 2.0f * PI * eta1;

        f32 sinTheta = eta0;
        f32 cosTheta = std::sqrtf(maximum(0.0f, 1.0f-sinTheta*sinTheta));

        wm.x_ = sinTheta * std::cosf(phi);
        wm.y_ = sinTheta * std::sinf(phi);
        wm.z_ = cosTheta;
        if(wm.z_<0.0f){
            wm.z_ = -wm.z_;
        }
    }

    f32 Diffuse::calcWeight(const Vector3& wi, const Vector3& wo) const
    {
        const Vector3 hr = normalizeChecked(wi + wo, Normal);
        return clamp01(hr.z_);
    }

    //-------------------------------------------------------------------
    //---
    //--- Isotropic Beckmann distribution
    //---
    //-------------------------------------------------------------------
    BeckmannIsotropic::BeckmannIsotropic(f32 alpha, f32 eta0, f32 eta1)
        :alpha_(alpha)
        ,eta0_(eta0)
        ,eta1_(eta1)
    {
        F0_ = clamp(absolute((eta0_-eta1_)/(eta0_+eta1_)), 0.0f, 2.0f);
    }

    Vector3 BeckmannIsotropic::evaluate(const Vector3& wi, const Vector3& wo) const
    {
        LASSERT(isEqual(wi.length(), 1.0f));
        LASSERT(isEqual(wo.length(), 1.0f));

        const Vector3 hr = normalizeChecked(wi + wo, Normal);
        f32 f = F(wi, hr, F0_);
        f32 gi = G1(wi);
        f32 go = G1(wo);
        f32 d = D(hr, Normal);
        f32 cosWiN = LocalCoordinate::cosTheta(wi);
        f32 cosWoN = LocalCoordinate::cosTheta(wo);
        f32 denom = maximum((4.0f * cosWiN * cosWoN), 1.0e-6f);
        return Vector3((f * gi * go * d) / denom);
    }

    void BeckmannIsotropic::sample(Vector3& wm, f32 eta0, f32 eta1) const
    {
        //Sample theta, phi
        f32 innerRoot = -(alpha_ * alpha_) * std::logf(1.0f - eta0);
        f32 tanTheta2 = innerRoot;
        f32 phi = 2.0f * PI * eta1;

        f32 cosTheta2 = 1.0f / (1.0f + tanTheta2);
        f32 cosTheta = std::sqrtf(cosTheta2);
        f32 sinTheta = std::sqrt(maximum(0.0f, 1.0f - cosTheta2));

        wm.x_ = sinTheta * std::cosf(phi);
        wm.y_ = sinTheta * std::sinf(phi);
        wm.z_ = cosTheta;
        if(cosTheta < 0.0f){
            wm.x_ = -wm.x_;
            wm.y_ = -wm.y_;
            wm.z_ = -wm.z_;
        }
    }

    f32 BeckmannIsotropic::D(const Vector3& m, const Vector3& n) const
    {
        f32 descr = dot(m, n);
        if(descr <= 0.0f){
            return 0.0f;
        }
        f32 alpha2 = alpha_ * alpha_;
        f32 cosTheta2 = maximum(LocalCoordinate::cosTheta2(m), 1.0e-6f);
        f32 tanTheta2 = (1.0f - cosTheta2) / cosTheta2;
        f32 denom = PI * cosTheta2 * cosTheta2 * alpha2;
        f32 expTanTheta2 = std::expf(-tanTheta2 / alpha2);
        return (F32_EPSILON <= denom)? expTanTheta2 / denom : 1.0f;
    }

    f32 BeckmannIsotropic::G1(const Vector3& v) const
    {
        f32 tanTheta2 = LocalCoordinate::tanTheta2(v);
        if(std::isinf(tanTheta2)){
            return 0.0f;
        }

        //Smith's approximation
        f32 a = (F32_EPSILON < alpha_)? 1.0f / (alpha_*std::sqrtf(tanTheta2)) : 2.0f;
        if(1.6f <= a){
            return 1.0f;
        }
        f32 a2 = a * a;
        return (3.535f * a + 2.181f * a2) / (1.0f + 2.276f * a + 2.577f * a2);
    }

    //-------------------------------------------------------------------
    //---
    //--- Isotropic GGX distribution
    //---
    //-------------------------------------------------------------------
    GGXIsotropic::GGXIsotropic(f32 alpha, f32 eta0, f32 eta1)
        :alpha_(alpha)
        ,eta0_(eta0)
        ,eta1_(eta1)
    {
        F0_ = clamp(absolute((eta0_-eta1_)/(eta0_+eta1_)), 0.0f, 2.0f);
    }

    Vector3 GGXIsotropic::evaluate(const Vector3& wi, const Vector3& wo) const
    {
        LASSERT(isEqual(wi.length(), 1.0f));
        LASSERT(isEqual(wo.length(), 1.0f));

        const Vector3 hr = normalizeChecked(wi + wo, Normal);
        f32 f = F(wi, hr, F0_);
        f32 gi = G1(wi);
        f32 go = G1(wo);
        f32 d = D(hr, Normal);
        f32 cosWiN = LocalCoordinate::cosTheta(wi);
        f32 cosWoN = LocalCoordinate::cosTheta(wo);
        f32 denom = maximum((4.0f * cosWiN * cosWoN), 1.0e-6f);
        return Vector3((f * gi * go * d) / denom);
    }

    void GGXIsotropic::sample(Vector3& wm, f32 eta0, f32 eta1) const
    {
        //Sample theta, phi

        f32 denom = eta0 * (alpha_*alpha_ - 1.0f) + 1.0f;
        f32 cosTheta = std::sqrtf((1.0f-eta0)/denom);
        f32 phi = 2.0f * PI * eta1;

        f32 sinTheta = std::sqrt(maximum(0.0f, 1.0f - cosTheta*cosTheta));

        wm.x_ = sinTheta * std::cosf(phi);
        wm.y_ = sinTheta * std::sinf(phi);
        wm.z_ = cosTheta;
        if(cosTheta < 0.0f){
            wm.x_ = -wm.x_;
            wm.y_ = -wm.y_;
            wm.z_ = -wm.z_;
        }
    }

    f32 GGXIsotropic::D(const Vector3& m, const Vector3& n) const
    {
        f32 descr = dot(m, n);
        if(descr <= 0.0f){
            return 0.0f;
        }

        f32 alpha2 = alpha_*alpha_;
        f32 cosTheta2 = LocalCoordinate::cosTheta2(m);
        f32 denom = (alpha2-1.0f)*cosTheta2 + 1.0f;
        return F32_EPSILON<denom? (alpha2/(denom*denom*PI)) : 0.0f;
    }

    f32 GGXIsotropic::G1(const Vector3& v) const
    {
        f32 tanTheta2 = LocalCoordinate::tanTheta2(v);
        if(std::isinf(tanTheta2)){
            return 0.0f;
        }

        //Smith's approximation
        f32 a = (F32_EPSILON < alpha_)? 1.0f / (alpha_*std::sqrtf(tanTheta2)) : 2.0f;
        if(1.6f <= a){
            return 1.0f;
        }
        f32 a2 = a * a;
        return (3.535f * a + 2.181f * a2) / (1.0f + 2.276f * a + 2.577f * a2);
    }
}
