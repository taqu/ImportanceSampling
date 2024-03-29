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

    //---------------------------------------------
    //---
    //--- Map
    //---
    //---------------------------------------------
    Vector3 Map::toUniformSphere(f32 x, f32 y)
    {
        f32 z = 1.0f - 2.0f * x;
        f32 r = std::sqrt(maximum(F32_EPSILON, (1.0f - z*z)));
        f32 phi = 2.0f*PI*y;
        f32 sinPhi = std::sinf(phi);
        f32 cosPhi = std::cosf(phi);
        return Vector3(r*cosPhi, r*sinPhi, z);
    }

    Vector3 Map::toUniformHemisphere(f32 x, f32 y)
    {
        f32 r = std::sqrt(maximum(F32_EPSILON, (1.0f - x*x)));
        f32 phi = 2.0f*PI*y;
        f32 sinPhi = std::sinf(phi);
        f32 cosPhi = std::cosf(phi);
        return Vector3(cosPhi*r, sinPhi*r, x);
    }

    Vector3 Map::toCosineHemisphere(f32 x, f32 y)
    {
        Vector2 p = toUniformDisk(x, y);
        f32 z = maximum(F32_EPSILON, std::sqrt(maximum(F32_EPSILON, (1.0f - p.x_*p.x_ - p.y_*p.y_))));
        return Vector3(p.x_, p.y_, z);
    }

    Vector3 Map::toUniformCone(f32 x, f32 y, f32 cosCutoff)
    {
        f32 cosTheta = (1.0f - x) + x*cosCutoff;
        f32 sinTheta = std::sqrt(maximum(F32_EPSILON, (1.0f - cosTheta*cosTheta)));
        f32 phi = 2.0f*PI*y;
        f32 sinPhi = std::sinf(phi);
        f32 cosPhi = std::cosf(phi);
        return Vector3(cosPhi*sinTheta, sinPhi*sinTheta, cosTheta);
    }

    Vector2 Map::toUniformDisk(f32 x, f32 y)
    {
        //from http://psgraphics.blogspot.ch/2011/01/improved-code-for-concentric-map.html

        f32 r0 = 2.0f*x - 1.0f;
        f32 r1 = 2.0f*y - 1.0f;
        f32 phi;
        f32 r;

        f32 absR0 = absolute(r0);
        f32 absR1 = absolute(r1);
        if(absR0<=F32_EPSILON && absR1<=F32_EPSILON){
            phi = r = 0.0f;

        }else if(absR1 < absR0){
            phi = (PI / 4.0f) * (r1/r0);
            r = r0;

        }else{
            r = r1;
            phi = (PI / 2.0f) - (r0/r1) * (PI / 4.0f);
        }

        return Vector2(r*std::cosf(phi), r*std::sinf(phi));
    }

    Vector2 Map::toUniformTriangle(f32 x, f32 y)
    {
        f32 a = std::sqrt(maximum(F32_EPSILON, (1.0f - x)));
        return Vector2(1.0f-a, a*y);
    }

    //---------------------------------------------
    //---
    //--- PDF
    //---
    //---------------------------------------------
    f32 PDF::cosineHemisphere(const Vector3& v)
    {
        return INV_PI * LocalCoordinate::cosTheta(v);
    }

    f32 PDF::uniformCone(f32 cosCutoff)
    {
        return INV_PI2 / (1.0f - cosCutoff);
    }

    //-------------------------------------------------------------------
    //---
    //--- Diffuse
    //---
    //-------------------------------------------------------------------
    Diffuse::Diffuse()
    {}

    Vector3 Diffuse::evaluate(const Vector3&, const Vector3&) const
    {
        return Vector3(INV_PI);
    }

    Vector3 Diffuse::sample(Vector3& wm, f32& pdf, const Vector3& wi, f32 eta0, f32 eta1) const
    {
        if(LocalCoordinate::cosTheta(wi)<=0.0f){
            pdf = 0.0f;
            return Vector3::Zero;
        }
        wm = Map::toCosineHemisphere(eta0, eta1);
        pdf = PDF::cosineHemisphere(wm);
        return Vector3::One;
    }

    //-------------------------------------------------------------------
    //---
    //--- Distribution
    //---
    //-------------------------------------------------------------------
    const Vector3 Distribution::Normal  = Vector3(0.0f, 0.0f, 1.0f);

    f32 Distribution::F(f32 cosTheta, f32 F0)
    {
        f32 r0 = F0*F0;
        f32 c = (1-cosTheta);
        f32 c2 = c*c;
        f32 c5 = c2*c2*c;
        return r0 + (1.0f-r0)*c5;
    }

    f32 Distribution::fresnelDielectric(f32 cosTheta, f32 eta)
    {
        if(isEqual(eta, 1.0f)){
            return 0.0f;
        }

        f32 scale = (0.0f<cosTheta)? 1.0f/eta : eta;
        f32 cosTheta2 = 1.0f - (1.0f - cosTheta*cosTheta)*(scale*scale);
        if(cosTheta<=F32_EPSILON){
            return 1.0f;
        }
        f32 cosThetaI = absolute(cosTheta);
        f32 cosThetaT = std::sqrtf(cosTheta2);
        f32 Rs = (cosThetaI - eta * cosThetaT)/(cosThetaI + eta*cosThetaT);
        f32 Rp = (eta*cosThetaI - cosThetaT)/(eta*cosThetaI + cosThetaT);
        return 0.5f * (Rs * Rs + Rp * Rp);
    }

    f32 Distribution::fresnelConductor(f32 cosTheta, f32 eta, f32 k)
    {
#if 1
        f32 cosTheta2 = cosTheta*cosTheta;
	    f32 sinTheta2 = 1.0f - cosTheta2;

        f32 eta2 = eta*eta;
        f32 k2 = k*k;
        f32 temp1 = eta2 - k2 - sinTheta2;
        f32 a2pb2 = std::sqrtf(temp1*temp1 + 4*k2*eta2);
        f32 a = std::sqrtf(0.5f * (a2pb2 + temp1));

        f32 term1 = a2pb2 + cosTheta2;
        f32 term2 = 2*a*cosTheta;

        f32 sum1 = term1 + term2;
        f32 Rs2 = (F32_EPSILON<sum1)? (term1 - term2) / sum1 : 1.0e3f;

        f32 term3 = a2pb2*cosTheta2 + sinTheta2*sinTheta2;
        f32 term4 = term2*sinTheta2;

        f32 sum2 = term3 + term4;
        f32 Rp2 = (F32_EPSILON<sum1)? Rs2 * (term3 - term4) / sum2 : 1.0e3f;

        return 0.5f * (Rp2 + Rs2);
#else
        f32 cosTheta2 = cosTheta*cosTheta;
        f32 tmpF = eta*eta + k*k;
        f32 tmp = tmpF * cosTheta2;
        f32 eta2 = eta*(2.0f*cosTheta);

        f32 Rp2 = (tmp - eta2 + 1.0f) / (tmp + eta2 + 1.0f); 
        f32 Rs2 = (tmpF - eta2 + cosTheta2) / (tmp + eta2 + cosTheta2);

        return 0.5f * (Rp2 + Rs2);
#endif
    }


    f32 Distribution::project2(f32 x, f32 y, const Vector3& v)
    {
        if(isEqual(x, y)){
            return x*x;
        }
        f32 sinTheta2 = LocalCoordinate::sinTheta2(v);
        if(sinTheta2<=F32_EPSILON){
            return x*x;
        }
        f32 invSinTheta2 = 1.0f/sinTheta2;
        f32 cosPhi2 = v.x_*v.x_*invSinTheta2;
        f32 sinPhi2 = v.y_*v.y_*invSinTheta2;
        return cosPhi2*x*x + sinPhi2*y*y;
    }

    Distribution::Distribution(f32 alphax, f32 alphay, f32 extEta, f32 intEta, f32 k)
        :alphax_(alphax)
        ,alphay_(alphay)
        ,extEta_(extEta)
        ,intEta_(intEta)
        ,k_(k)
    {
        f32 invExtEta = 1.0f/extEta_;
        eta_ = intEta_*invExtEta;
        if(isEqual(eta_, 1.0f)){
            eta_ += 1.0e-3f;
        }
        k_ *= invExtEta;
        F0_ = clamp(absolute((extEta - intEta) / (extEta + intEta)), 0.0f, 2.0f);
    }

    Distribution::~Distribution()
    {}

    f32 Distribution::fresnel(f32 cosTheta) const
    {
        return F(cosTheta, F0_);
    }

    f32 Distribution::fresnelDielectric(f32 cosTheta) const
    {
        return fresnelDielectric(cosTheta, eta_);
    }
    f32 Distribution::fresnelConductor(f32 cosTheta) const
    {
        return fresnelConductor(cosTheta, eta_, k_);
    }

    //-------------------------------------------------------------------
    //---
    //--- Isotropic Beckmann distribution
    //---
    //-------------------------------------------------------------------
    BeckmannIsotropic::BeckmannIsotropic(f32 alphax, f32 alphay, f32 extEta, f32 intEta, f32 k)
        :Distribution(alphax, alphay, extEta, intEta, k)
    {}

    Vector3 BeckmannIsotropic::evaluate(const Vector3& wi, const Vector3& wo) const
    {
        LASSERT(isEqual(wi.length(), 1.0f));
        LASSERT(isEqual(wo.length(), 1.0f));

        const Vector3 hr = normalizeChecked(wi + wo, Normal);
        f32 f = fresnel(dot(wi,hr));
        f32 gi = G1(wi, hr);
        f32 go = G1(wo, hr);
        f32 d = D(hr, Normal);
        f32 cosWiN = LocalCoordinate::cosTheta(wi);
        f32 cosWoN = LocalCoordinate::cosTheta(wo);
        f32 denom = maximum((4.0f * cosWiN * cosWoN), 1.0e-6f);
        return Vector3((f * gi * go * d) / denom);
    }

    void BeckmannIsotropic::sample(Vector3& m, f32& pdf, const Vector3& wi, f32 eta0, f32 eta1) const
    {
        //Sample theta, phi
        f32 innerRoot = -(alphax_ * alphax_) * std::logf(1.0f - eta0);
        f32 tanTheta2 = innerRoot;
        f32 phi = 2.0f * PI * eta1;

        f32 cosTheta2 = 1.0f / (1.0f + tanTheta2);
        f32 cosTheta = std::sqrtf(cosTheta2);
        f32 sinTheta = std::sqrt(maximum(0.0f, 1.0f - cosTheta2));

        m.x_ = sinTheta * std::cosf(phi);
        m.y_ = sinTheta * std::sinf(phi);
        m.z_ = cosTheta;
        if(cosTheta < 0.0f){
            m.x_ = -m.x_;
            m.y_ = -m.y_;
            m.z_ = -m.z_;
        }

        pdf = D(m) * LocalCoordinate::cosTheta(m);
    }

    f32 BeckmannIsotropic::calcWeight(const Vector3& wi, const Vector3& wo, const Vector3& m, f32 pdf) const
    {
        return minimum(D(m) * G1(wi, m) * dot(wi, m)/(pdf * LocalCoordinate::cosTheta(wi)), 1.0e5f);
    }

    f32 BeckmannIsotropic::D(const Vector3& m, const Vector3& n) const
    {
        f32 descr = dot(m, n);
        if(descr <= 0.0f){
            return 0.0f;
        }
        f32 alpha2 = maximum(alphax_ * alphax_, 1.0e-6f);
        f32 cosTheta2 = maximum(LocalCoordinate::cosTheta2(m), 1.0e-6f);
        f32 tanTheta2 = (1.0f - cosTheta2) / cosTheta2;
        f32 denom = PI * cosTheta2 * cosTheta2 * alpha2;
        f32 expTanTheta2 = std::expf(-tanTheta2 / alpha2);
        return (F32_EPSILON <= denom)? expTanTheta2 / denom : 1.0f;
    }

    f32 BeckmannIsotropic::G1(const Vector3& v, const Vector3& m) const
    {
        if(dot(v, m)*LocalCoordinate::cosTheta(v)<=F32_EPSILON){
            return 0.0f;
        }

        f32 tanTheta = absolute(LocalCoordinate::tanTheta(v));
        if(tanTheta<=F32_EPSILON){
            return 1.0f;
        }

        f32 a = (F32_EPSILON < alphax_)? 1.0f / (alphax_*tanTheta) : 2.0f;
        if(1.6f <= a){
            return 1.0f;
        }
        f32 a2 = a * a;
        return (3.535f * a + 2.181f * a2) / (1.0f + 2.276f * a + 2.577f * a2);
    }

    f32 BeckmannIsotropic::D(const Vector3& m) const
    {
        f32 cosTheta = LocalCoordinate::cosTheta(m);
        if(cosTheta<=F32_EPSILON){
            return 0.0f;
        }

        f32 alpha2 = (F32_EPSILON < alphax_)? 1.0f/(alphax_ * alphax_) : 1.0f;
        f32 cosTheta2 = 1.0f/maximum(LocalCoordinate::cosTheta2(m), 1.0e-6f);
        f32 beckmann = ((m.x_*m.x_) * alpha2 + (m.y_*m.y_) * alpha2) * cosTheta2;

        f32 result = std::expf(-beckmann) * alpha2 * cosTheta2 * INV_PI;
        return F32_EPSILON<(result*cosTheta)? result : 0.0f;
    }

    //-------------------------------------------------------------------
    //---
    //--- Isotropic GGX distribution
    //---
    //-------------------------------------------------------------------
    GGXIsotropic::GGXIsotropic(f32 alphax, f32 alphay, f32 extEta, f32 intEta, f32 k)
        :Distribution(alphax, alphay, extEta, intEta, k)
    {}

    Vector3 GGXIsotropic::evaluate(const Vector3& wi, const Vector3& wo) const
    {
        LASSERT(isEqual(wi.length(), 1.0f));
        LASSERT(isEqual(wo.length(), 1.0f));

        const Vector3 hr = normalizeChecked(wi + wo, Normal);
        f32 f = fresnel(dot(wi, hr));
        f32 gi = G1(wi, hr);
        f32 go = G1(wo, hr);
        f32 d = D(hr, Normal);
        f32 cosWiN = LocalCoordinate::cosTheta(wi);
        f32 cosWoN = LocalCoordinate::cosTheta(wo);
        f32 denom = maximum((4.0f * cosWiN * cosWoN), 1.0e-6f);
        return Vector3((f * gi * go * d) / denom);
    }

    void GGXIsotropic::sample(Vector3& m, f32& pdf, const Vector3&, f32 eta0, f32 eta1) const
    {
        //Sample theta, phi

        f32 denom = eta0 * (alphax_*alphax_ - 1.0f) + 1.0f;
        f32 cosTheta = std::sqrtf((1.0f-eta0)/denom);
        f32 phi = 2.0f * PI * eta1;

        f32 sinTheta = std::sqrt(maximum(0.0f, 1.0f - cosTheta*cosTheta));

        m.x_ = sinTheta * std::cosf(phi);
        m.y_ = sinTheta * std::sinf(phi);
        m.z_ = cosTheta;
        if(cosTheta < 0.0f){
            m.x_ = -m.x_;
            m.y_ = -m.y_;
            m.z_ = -m.z_;
        }

        pdf = D(m) * LocalCoordinate::cosTheta(m);
    }

    f32 GGXIsotropic::calcWeight(const Vector3& wi, const Vector3& wo, const Vector3& m, f32 pdf) const
    {
        return minimum(D(m) * G1(wi, m) * dot(wi, m)/(pdf * LocalCoordinate::cosTheta(wi)), 1.0e5f);
    }

    f32 GGXIsotropic::D(const Vector3& m, const Vector3& n) const
    {
        f32 descr = dot(m, n);
        if(descr <= 0.0f){
            return 0.0f;
        }

        f32 alpha2 = alphax_*alphax_;
        f32 cosTheta2 = LocalCoordinate::cosTheta2(m);
        f32 denom = (alpha2-1.0f)*cosTheta2 + 1.0f;
        return F32_EPSILON<denom? (alpha2/(denom*denom*PI)) : 0.0f;
    }

    f32 GGXIsotropic::G1(const Vector3& v, const Vector3& m) const
    {
        if(dot(v, m)*LocalCoordinate::cosTheta(v)<=F32_EPSILON){
            return 0.0f;
        }

        f32 tanTheta = absolute(LocalCoordinate::tanTheta(v));
        if(tanTheta<=F32_EPSILON){
            return 1.0f;
        }

        f32 tanTheta2 = tanTheta*tanTheta;
        return false == std::isinf(tanTheta2)? 2.0f/(1.0f + std::sqrtf(1.0f + alphax_*alphax_*tanTheta2)) : 0.0f;
    }

    f32 GGXIsotropic::D(const Vector3& m) const
    {
        f32 cosTheta = LocalCoordinate::cosTheta(m);
        if(cosTheta<=F32_EPSILON){
            return 0.0f;
        }

        f32 alpha2 = 1.0f/maximum(alphax_ * alphax_, 1.0e-6f);
        f32 cosTheta2 = maximum(LocalCoordinate::cosTheta2(m), 1.0e-6f);
        f32 beckmann = ((m.x_*m.x_) * alpha2 + (m.y_*m.y_) * alpha2);
        f32 denom = beckmann + cosTheta2;
        f32 result = (INV_PI*alpha2)/(denom * denom);
        return F32_EPSILON<(result*cosTheta)? result : 0.0f;
    }

    //-------------------------------------------------------------------
    //---
    //--- Anisotropic GGX distribution
    //---
    //-------------------------------------------------------------------
    GGXAnisotropicVND::GGXAnisotropicVND(f32 alphax, f32 alphay, f32 extEta, f32 intEta, f32 k)
        :Distribution(alphax, alphay, extEta, intEta, k)
    {}

    Vector3 GGXAnisotropicVND::evaluate(const Vector3& wi, const Vector3& wo) const
    {
        LASSERT(isEqual(wi.length(), 1.0f));
        LASSERT(isEqual(wo.length(), 1.0f));

        const Vector3 hr = normalizeChecked(wi + wo, Normal);
        f32 f = fresnel(dot(wi, hr));
        f32 gi = G1(wi, hr);
        f32 go = G1(wo, hr);
        f32 d = D(hr, Normal);
        f32 cosWiN = LocalCoordinate::cosTheta(wi);
        f32 cosWoN = LocalCoordinate::cosTheta(wo);
        f32 denom = maximum((4.0f * cosWiN * cosWoN), 1.0e-6f);
        return Vector3((f * gi * go * d) / denom);
    }

    void GGXAnisotropicVND::sample(Vector3& m, f32& pdf, const Vector3& wi, f32 eta0, f32 eta1) const
    {
        //1. streached omega i
        Vector3 omegai;
        omegai.x_ = alphax_ * wi.x_;
        omegai.y_ = alphay_ * wi.y_;
        omegai.z_ = wi.z_;
        omegai = normalize(omegai);

        //get polar coordinates of omega i
        f32 theta, phi;
        if(omegai.z_<0.999f){
            theta = std::acosf(omegai.z_);
            phi = std::atan2f(omegai.y_, omegai.x_);
        }else{
            theta = phi = 0.0f;
        }

        //2. sample P22
        f32 slopex, slopey;
        sample11(slopex, slopey, theta, eta0, eta1);
        if(std::isnan(slopex) || std::isnan(slopey)){
            m = Normal;
            pdf = 0.0f;
            return;
        }

        //3. rotate
        f32 cosPhi = std::cosf(phi);
        f32 sinPhi = std::sinf(phi);
        f32 tmpx = cosPhi*slopex - sinPhi*slopey;
        slopey = sinPhi*slopex + cosPhi*slopey;
        slopex = tmpx;

        //4. unstretch
        slopex *= alphax_;
        slopey *= alphay_;

        //5. compute normal
        f32 omegam = 1.0f/std::sqrtf(1.0f + slopex*slopex + slopey*slopey);
        m.x_ = -slopex * omegam;
        m.y_ = -slopey * omegam;
        m.z_ = omegam;
        if(std::isnan(omegam)){
            m = Normal;
            pdf = 0.0f;
            return;
        }
        //m = normalize(m);

        // pdf
        f32 cosTheta = absolute(LocalCoordinate::cosTheta(wi));
        if(cosTheta<F32_EPSILON*2){
            pdf = 0.0f;
        } else{
            pdf = G1(wi, m) * absolute(dot(wi, m)) * D(m)/cosTheta;
        }
    }

    f32 GGXAnisotropicVND::calcWeight(const Vector3& wi, const Vector3& wo, const Vector3& m, f32 pdf) const
    {
        return G1(wi, m);
    }

    f32 GGXAnisotropicVND::D(const Vector3& m, const Vector3& n) const
    {
        f32 descr = dot(m, n);
        if(descr <= 0.0f){
            return 0.0f;
        }

        f32 alphax2 = alphax_*alphax_;
        f32 alphay2 = alphay_*alphay_;
        f32 cosTheta2 = LocalCoordinate::cosTheta2(m);
        f32 sinTheta2 = 1.0f - cosTheta2;

        f32 cosPhi2 = m.x_*m.x_;
        f32 sinPhi2 = m.y_*m.y_;

        f32 denom = cosTheta2 + (cosPhi2/alphax2 + sinPhi2/alphay2) * sinTheta2;
        return F32_EPSILON<denom? 1.0f/(PI*alphax_*alphay_*denom*denom) : 0.0f;
    }

    f32 GGXAnisotropicVND::G1(const Vector3& v, const Vector3& m) const
    {
        if(dot(v, m)*LocalCoordinate::cosTheta(v)<=F32_EPSILON){
            return 0.0f;
        }

        f32 tanTheta = absolute(LocalCoordinate::tanTheta(v));
        if(tanTheta<=F32_EPSILON){
            return 1.0f;
        }

        f32 tanTheta2 = tanTheta*tanTheta;
        f32 alpha2 = project2(alphax_, alphay_, v);
        return false == std::isinf(tanTheta2)? 2.0f/(1.0f + std::sqrtf(1.0f + alpha2*tanTheta2)) : 0.0f;
    }

    f32 GGXAnisotropicVND::D(const Vector3& m) const
    {
        f32 cosTheta = LocalCoordinate::cosTheta(m);
        if(cosTheta<=F32_EPSILON){
            return 0.0f;
        }

        f32 cosTheta2 = cosTheta*cosTheta;
        f32 beckmann = ((m.x_*m.x_) / (alphax_*alphax_) + (m.y_*m.y_)/(alphay_*alphay_));
        f32 denom = beckmann + cosTheta2;
        f32 result = 1.0f/(PI * alphax_ * alphay_ * denom * denom);
        return F32_EPSILON<(result*cosTheta)? result : 0.0f;
    }
            
    void GGXAnisotropicVND::sample11(f32& slopex, f32& slopey, f32 thetai, f32 eta0, f32 eta1) const
    {
        if(thetai<1.0e-4f){
            f32 r = std::sqrt(eta0/(1.0f-eta0));
            f32 phi = PI2*eta1;
            slopex = r * std::cosf(phi);
            slopey = r * std::sinf(phi);
            return;
        }
        //precomputations
        f32 tanTheta = std::tanf(thetai);
        f32 a = 1.0f/tanTheta;
        f32 G1 = 2.0f/(1.0f + std::sqrtf(1.0f + 1.0f/(a*a)));

        //sample slope x
        f32 A = 2.0f*eta0/G1 - 1.0f;
        if((absolute(A)-1.0f)<=F32_EPSILON){
            A = (A<0.0f)? A+F32_EPSILON : A-F32_EPSILON;
        }

        f32 tmp = 1.0f/(A*A - 1.0f);
        f32 B = tanTheta;
        f32 D = std::sqrtf(B*B*tmp*tmp - (A*A - B*B)*tmp);
        f32 slopex_0 = B*tmp - D;
        f32 slopex_1 = B*tmp + D;
        slopex = (A<0.0f || (1.0f/tanTheta)<slopex_1)? slopex_0 : slopex_1;

        //sample slope y
        s32 S;
        if(0.5f<eta1){
            S = 1;
            eta1 = 2.0f*(eta1 - 0.5f);
        }else{
            S = -1;
            eta1 = 2.0f*(0.5f - eta1);
        }
        f32 z = (eta1*(eta1*(eta1*0.27385f - 0.73369f) + 0.46341f)) / (eta1*(eta1*(eta1*0.093073f + 0.309420f)-1.0f)+0.597999f);
        slopey = S * z * std::sqrtf(1.0f+slopex*slopex);
    }

    //-------------------------------------------------------------------
    //---
    //--- Sampling on anisotropic GGX VNDF
    //---
    //-------------------------------------------------------------------
    GGXAnisotropicElipsoidVND::GGXAnisotropicElipsoidVND(f32 alphax, f32 alphay, f32 extEta, f32 intEta, f32 k)
        :GGXAnisotropicVND(alphax, alphay, extEta, intEta, k)
    {}

    Vector3 GGXAnisotropicElipsoidVND::evaluate(const Vector3& wi, const Vector3& wo) const
    {
        LASSERT(isEqual(wi.length(), 1.0f));
        LASSERT(isEqual(wo.length(), 1.0f));

        const Vector3 hr = normalizeChecked(wi + wo, Normal);
        f32 f = fresnel(dot(wi, hr));
        f32 gi = G1(wi, hr);
        f32 go = G1(wo, hr);
        f32 d = D(hr, Normal);
        f32 cosWiN = LocalCoordinate::cosTheta(wi);
        f32 cosWoN = LocalCoordinate::cosTheta(wo);
        f32 denom = maximum((4.0f * cosWiN * cosWoN), 1.0e-6f);
        return Vector3((f * gi * go * d) / denom);
    }

    void GGXAnisotropicElipsoidVND::sample(Vector3& m, f32& pdf, const Vector3& wi, f32 eta0, f32 eta1) const
    {
        //1. Transform the view direction to the hemisphere configuration
        Vector3 Vh = normalize(Vector3(alphax_*wi.x_, alphay_*wi.y_, wi.z_));

        //2. Construct orthonormal bais
        Vector3 T1 = (Vh.z_<0.999f)? normalize(cross(Vector3::Forward, Vh)) : Vector3::Right;
        Vector3 T2 = cross(Vh, T1);

        //3. Make parameterization of the projected area
        f32 r = std::sqrtf(eta0);
        f32 phi = PI2 * eta1;
        f32 t1 = r * std::cosf(phi);
        f32 t2 = r * std::sinf(phi);
        f32 s = 0.5f * (1.0f + Vh.z_);
        t2 = (1.0f - s)*std::sqrtf(1.0f-t1*t1) + s*t2;

        //4. Reproject onto hemisphere
        Vector3 Nh = t1*T1 + t2*T2 + std::sqrtf(maximum(0.0f, 1.0f-t1*t1-t2*t2)) * Vh;

        //5. Transform the normal back to the elipsoid configuration
        m = normalize(Vector3(alphax_*Nh.x_, alphay_*Nh.y_, maximum(0.0f, Nh.z_)));

        // pdf
        f32 cosTheta = absolute(LocalCoordinate::cosTheta(wi));
        if(cosTheta<F32_EPSILON*2){
            pdf = 0.0f;
        } else{
            pdf = G1(wi, m) * absolute(dot(wi, m)) * D(m)/cosTheta;
        }
    }
}
