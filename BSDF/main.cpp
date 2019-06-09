#include <stdio.h>
#include <array>
#include <limits>
#include "BSDF.h"
#include "Coordinate.h"

using namespace lrender;

template<class T>
void test(f32 alpha, f32 eta0, f32 eta1, const char* distribution, const char* samples)
{
    T dist(alpha, eta0, eta1);
    FILE* file= fopen(distribution, "wb");
    if(NULL == file){
        return;
    }
    const lrender::s32 Resolution = 128;
    const lrender::Vector3 n = {0.0f, 0.0f, 1.0f};
    for(lrender::s32 i=0; i<=Resolution; ++i){
        lrender::f32 theta = (lrender::PI*0.5f)*(static_cast<lrender::f32>(i)/Resolution);
        //theta = (lrender::PI*0.5f)*(static_cast<lrender::f32>(Resolution)/Resolution);
        lrender::Vector3 wi = {std::sinf(theta), 0.0f, std::cosf(theta)};
        lrender::Vector3 bsdf = dist.evaluate(wi, wi);
        fprintf(file, "%f,%f\n", theta, bsdf.x_);
    }
    fclose(file);

    lrender::Xoshiro128Plus random(12345);

    file= fopen(samples, "wb");
    if(NULL == file){
        return;
    }

    const lrender::s32 Samples = 1024*1024;
    lrender::f32 count[90] = {};
    for(lrender::s32 i=0; i<=Samples; ++i){
        lrender::f32 eta0 = random.frand2();
        lrender::f32 eta1 = random.frand2();
        lrender::Vector3 wm;
        dist.sample(wm, eta0, eta1);

        f32 cosTheta = dot(wm, BSDF::Normal);
        lrender::s32 deg = static_cast<lrender::s32>(std::acos(cosTheta)*RAD_TO_DEG);
        if(90<=deg){
            deg = 89;
        }
        lrender::f32 w = dist.calcWeight(wm, wm);
        count[deg] += 1.0f;//w;
    }
    for(lrender::s32 i=0; i<90; ++i){
        fprintf(file, "%f\n", count[i]);
    }
    fclose(file);
}

f32 sRGBToLinear(f32 x)
{
    if(0.0f<=x && x<=0.04045f){
        return x/12.92f;
    }else if(0.04045f<x && x<=1.0f){
        return std::powf((x+0.055f)/1.055f, 2.4f);
    }else{
        return x;
    }
}

f32 linearToSRGB(f32 x)
{
    if(0.0f<=x && x<=0.0031308f){
        return x*12.92f;
    }else if(0.0031308f<x<=1.0f){
        return std::powf(1.055f*x, 1.0f/2.4f) - 0.055f;
    }else{
        return x;
    }
}

u8 toU8(f32 x)
{
    s32 i = static_cast<s32>(x*255+0.5f);
    return i<256? static_cast<u8>(i) : 255;
}
    
enum Type
{
    DIFFUSE,
    ROUGH_CONDUCTOR,
};

struct TSphere
{
    f32 radius_;
    Vector3 position_;
    Vector3 emission_;
    Vector3 color_;
    Type type_;

    TSphere(f32 radius, const Vector3& position, const Vector3& emission, const Vector3& color, Type type)
        :radius_(radius)
        ,position_(position)
        ,emission_(emission)
        ,color_(color)
        ,type_(type)
    {}

    // returns distance, (std::numeric_limits<f32>::max)() if nohit
    f32 intersect(const Ray& ray) const
    {
        Vector3 m = ray.origin_ - position_;

        f32 b = dot(m, ray.direction_);
        f32 c = dot(m, m) - radius_ * radius_;

        if(0.0f<c && 0.0f<b){
            return (std::numeric_limits<f32>::max)();
        }

        f32 discr = b*b - c;
        if(discr < 0.0f){
            return (std::numeric_limits<f32>::max)();
        }

        discr = std::sqrtf(discr);
        b = -b;

        f32 tmin = b - discr;
        f32 tmax = b + discr;
        return LRENDER_RAY_EPSILON<tmin? tmin : LRENDER_RAY_EPSILON<tmax? tmax : (std::numeric_limits<f32>::max)();
    }
};

std::array<TSphere, 9> spheres{//Scene: radius, position, emission, color, material
    TSphere(1e1, Vector3( 1e1+50e-4, 40.8e-4, 0.0f), Vector3::Zero,Vector3(.75,.25,.25),DIFFUSE),//Left
    TSphere(1e1, Vector3(-1e1-50e-4, 40.8e-4, 0.0f), Vector3::Zero,Vector3(.25,.25,.75),DIFFUSE),//Rght
    TSphere(1e1, Vector3(0.0f, 40.8e-4,  1e1+100e-4), Vector3::Zero,Vector3(.25,.25,.25), DIFFUSE),//Back
    TSphere(1e1, Vector3(0.0f, 40.8e-4, -1e1-50e-4), Vector3::Zero,Vector3(.75,.75,.25), DIFFUSE),//Frnt
    TSphere(1e1, Vector3(0.0f,  -1e1, 0.0f),    Vector3::Zero,Vector3(.75,.75,.75),DIFFUSE),//Botm
    TSphere(1e1, Vector3(0.0f, 1e1+81.6e-4,0.0f),Vector3::Zero,Vector3(.75,.75,.75),DIFFUSE),//Top
    TSphere(16.5e-4f,Vector3(-30.0e-4f, 16.5e-4f, -30e-4f), Vector3::Zero,Vector3(1,1,1)*.999, ROUGH_CONDUCTOR),//Mirr
    TSphere(16.5e-4f,Vector3(25e-4f,16.5e-4f,-10e-4f), Vector3::Zero,Vector3(1,1,1)*.999, DIFFUSE),//Glas
    TSphere(16.0e-4f, Vector3(0.0f,12.0e-4f+81.6e-4, 0.0f), Vector3(12,12,12), Vector3::Zero, DIFFUSE), //Light
};

struct Intersection
{
    s32 id_; //id of an intersected object
    f32 t_; //distance to intersection
    Vector3 position_;
    Vector3 normal_;
    Vector3 binormal0_;
    Vector3 binormal1_;

    Vector3 worldToLocal(const Vector3& v) const
    {
        return normalize(Vector3(dot(v, binormal0_), dot(v, binormal1_), dot(v, normal_)));
    }

    Vector3 Intersection::localToWorld(const Vector3& v) const
    {
        return Vector3(
            binormal0_.x_ * v.x_ + binormal1_.x_ * v.y_ + normal_.x_ * v.z_,
            binormal0_.y_ * v.x_ + binormal1_.y_ * v.y_ + normal_.y_ * v.z_,
            binormal0_.z_ * v.x_ + binormal1_.z_ * v.y_ + normal_.z_ * v.z_);
    }
};

bool intersect(const Ray& ray, Intersection& intersection)
{
    intersection.t_ = (std::numeric_limits<f32>::max)();
    for(s32 i=0; i<spheres.size(); ++i){
        f32 d = spheres[i].intersect(ray);
        if(d<intersection.t_){
            intersection.t_ = d;
            intersection.id_ = i;
        }
    }
    return intersection.t_<(std::numeric_limits<f32>::max)();
}

void calcIntersection(Intersection& intersection, const Ray& ray)
{
    intersection.position_ = ray.origin_ + intersection.t_*ray.direction_;
    intersection.normal_ = normalizeChecked(intersection.position_-spheres[intersection.id_].position_, Vector3::Zero);

    orthonormalBasis(intersection.binormal0_, intersection.binormal1_, intersection.normal_);
}


template<class T>
Vector3 radiance(Ray ray, s32 maxDepth, f32 roughness, lrender::Xoshiro128Plus& random)
{
    Vector3 L = Vector3::Zero;
    Vector3 beta = Vector3::One;
    f32 pdf = 1.0f;
    Intersection intersection;
    for(s32 depth = 0; depth<maxDepth; ++depth){
        f32 t = 0.0f; //distance to intersection
        s32 id = 0;   //id of an intersected object
        if(!intersect(ray, intersection)) {
            break;
        }
        calcIntersection(intersection, ray);
        const TSphere& sphere = spheres[intersection.id_];
        L = L + beta * sphere.emission_ / pdf;

        Vector3 wow = -ray.direction_;
        Vector3 wo = intersection.worldToLocal(wow);
        Vector3 n = intersection.worldToLocal(intersection.normal_);
        f32 d = dot(wow, intersection.normal_);
        Vector3 wiw;

        switch(sphere.type_) {
        case DIFFUSE: {
            Diffuse diffuse;
            f32 eta0 = random.frand2();
            f32 eta1 = random.frand2();
            Vector3 wi;
            diffuse.sample(wi, eta0, eta1);
            f32 bsdfPdf = LocalCoordinate::isSameHemisphere(wo, wi) ? LocalCoordinate::absCosTheta(wi)*INV_PI : 0.0f;
            Vector3 f = diffuse.evaluate(wi, wo) * sphere.color_;
            if(f.isEqual(Vector3::Zero) || bsdfPdf <= LRENDER_PDF_EPSILON) {
                return L;
            }
            wiw = intersection.localToWorld(wi);
            beta *= f * clamp01(dot(wiw, intersection.normal_))/bsdfPdf;
        } break;
        case ROUGH_CONDUCTOR:{
#if 1
            T distribution(roughness, 1.0f, 1.333f);
            f32 eta0 = random.frand2();
            f32 eta1 = random.frand2();
            Vector3 wi;
            distribution.sample(wi, eta0,eta1);
            f32 bsdfPdf = LocalCoordinate::isSameHemisphere(wo, wi)? distribution.calcWeight(wi, wo) : 0.0f;
            if(bsdfPdf <= LRENDER_PDF_EPSILON) {
                return L;
            }
#else

            f32 eta0 = random.frand2();
            f32 eta1 = random.frand2();

            f32 theta = PI * eta0;
            f32 phi = 2.0f * PI * eta1;
            f32 sinTheta = std::sinf(theta);
            f32 cosTheta = std::cosf(theta);

            Vector3 wi;
            wi.x_ = sinTheta * std::cosf(phi);
            wi.y_ = sinTheta * std::sinf(phi);
            wi.z_ = cosTheta;
            if(wi.z_<0.0f){
                wi.z_ = -wi.z_;
            }
            wi = normalize(wi);

            BeckmannIsotropic beckmann(0.2f, 1.0f, 1.333f);
            f32 bsdfPdf = LocalCoordinate::isSameHemisphere(wo, wi)? beckmann.evaluate(wi, wo).x_ : 0.0f;
            if(bsdfPdf <= LRENDER_PDF_EPSILON) {
                return L;
            }
#endif
            wiw = intersection.localToWorld(wi);
            beta *= sphere.color_ * bsdfPdf;
        }break;
        default:
            return L;
        }

        ray.origin_= intersection.position_;//muladd(LRENDER_RAY_EPSILON, intersection.normal_, intersection.position_);
        ray.direction_ = wiw;

        //Russian roulette
        if(6 <= depth) {
            f32 continueProbability = minimum(beta.length(), 0.9f);
            if(continueProbability <= random.frand2()) {
                break;
            }
            beta /= continueProbability;
        }
    }
    return L;
}

template<class T>
void render(const char* name, s32 width, s32 height, s32 spp, u32 seed, f32 roughness)
{
    lrender::Xoshiro128Plus random(seed);

    static const s32 Width = width;
    static const s32 Height = height;

    f32 aspect = static_cast<f32>(Width)/Height;
    f32 angle = 65.0f*DEG_TO_RAD;
    f32 fovx = std::tanf(angle*0.5f);
    f32 fovy = fovx/aspect;

    Ray camera(Vector3(0.0f, 40.8e-4f, 120e-4), normalize(Vector3(0, -0.042612f, -1)));
    Vector3 cx = Vector3(fovx, 0.0, 0.0);
    Vector3 cy = normalize(cross(cx, camera.direction_)) * fovy;
    auto_array_ptr<Vector3> screen(GFX_NEW Vector3[Width*Height]);
    memset(screen, 0, sizeof(Vector3)*Width*Height);

    Timer<> timer;
    timer.start();
    f32 invWidth2 = 2.0f/Width;
    f32 invHeight2 = 2.0f/Height;
    f32 invSpp = 1.0f/spp;

    for(s32 y=0; y<Height; ++y){
        s32 row = (Height-y-1)*Width;
        for(s32 x=0; x<Width; ++x){
            Vector3 r = Vector3::Zero;
            for(s32 s = 0; s<spp; ++s){
                f32 rx = random.frand2();
                f32 ry = random.frand2();
                f32 sx = (x + rx)*invWidth2 - 1.0f;
                f32 sy = (y + ry)*invHeight2 - 1.0f;

                Vector3 d = normalize(cx*sx + cy*sy + camera.direction_);
                Ray ray;
                ray.origin_ = camera.origin_ + d*30e-4;
                ray.direction_ = d;
                r += radiance<T>(ray, 16, roughness, random);
            }
            r *= invSpp;
            screen[row+x] += Vector3(clamp01(r.x_), clamp01(r.y_), clamp01(r.z_));
        }
    }

    timer.stop();
    printf("Time: %f sec\n", timer.getAverage());

    FILE *f = fopen(name, "w");         // Write image to PPM file.
    fprintf(f, "P3\n%d %d\n%d\n", Width, Height, 255);
    for(s32 i = 0; i<(Width*Height); ++i){
        u8 r = toU8(linearToSRGB(screen[i].x_));
        u8 g = toU8(linearToSRGB(screen[i].y_));
        u8 b = toU8(linearToSRGB(screen[i].z_));
        fprintf(f, "%d %d %d ", r,g,b);
    }
    fclose(f);
}

int main(int argc, char** argv)
{
    //test<BeckmannIsotropic>(0.5f, 1.0f, 1.33f, "beckmann_out.csv", "beckmann_sample.csv");
    //test<GGXIsotropic>(0.5f, 1.0f, 1.33f, "ggx_out.csv", "ggx_sample.csv");

    render<BeckmannIsotropic>("beckmann.ppm", 1024, 768, 4096, 12345, 0.2f);
    render<GGXIsotropic>("ggx.ppm", 1024, 768, 4096, 12345, 0.2f);
    return 0;
}
