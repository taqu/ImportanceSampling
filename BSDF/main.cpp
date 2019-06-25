#include <stdio.h>
#include <array>
#include <limits>
#include "BSDF.h"
#include "Coordinate.h"

using namespace lrender;

struct WeightedSample
{
    f32 calcAverage()
    {
        return 0<count_? weight_/count_ : 0.0f;
    }

    int count_;
    float weight_;
    float pdf_;
};

template<class T>
void test(f32 alpha, u32 seed, const char* distribution)
{
    lrender::Xoshiro128Plus random(seed);
    const lrender::s32 Samples = 1024*1024;
    WeightedSample count[90] = {};

    BSDF<T> bsdf(alpha, alpha, 1.0f, 1.8f, 0.0f);
    for(s32 i=0; i<=Samples; ++i){
        Vector3 wi = normalize(Vector3(1.0f, 0.0f, 1.0f));
        f32 eta0 = random.frand2();
        f32 eta1 = random.frand2();
        Vector3 wo;
        f32 bsdfPdf;
        Vector3 weight = bsdf.sample(wo, bsdfPdf, wi, eta0, eta1);
        f32 cosTheta = dot(wi, wo);
        if(cosTheta<=0.0f){
            continue;
        }
        Vector3 wm = normalizeChecked(wo+wi, Vector3::Forward);
        cosTheta = dot(wo, wm);
        s32 deg = static_cast<lrender::s32>(std::acos(clamp01(cosTheta))*RAD_TO_DEG);
        if(90<=deg){
            deg = 89;
        }
        count[deg].count_ = 1;
        count[deg].weight_ += weight.x_;
        count[deg].pdf_ += bsdfPdf;
    }

    FILE* file = fopen(distribution, "wb");
    if(LRENDER_NULL == file){
        return;
    }
    for(s32 i=0; i<90; ++i){
        fprintf(file, "%f,%f\n", count[i].weight_, count[i].pdf_);
    }
    fclose(file);
}

void fresnel()
{
    FILE* file= fopen("fresnel.csv", "wb");
    if(NULL == file){
        return;
    }
    const lrender::s32 Resolution = 128;
    const lrender::Vector3 n = {0.0f, 0.0f, 1.0f};
    for(lrender::s32 i=0; i<=Resolution; ++i){
        f32 cosTheta = static_cast<lrender::f32>(i)/Resolution;
        f32 f = Distribution::fresnelDielectric(cosTheta, 1.5f);
        fprintf(file, "%f,%f\n", cosTheta, f);
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
    }else if(0.0031308f<x && x<=1.0f){
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
    
enum MaterialType
{
    EMITTER,
    DIFFUSE,
    ROUGH_CONDUCTOR,
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

struct Shape
{
    enum Type
    {
        Type_Sphere,
        Type_Plane,
    };

    Type shapeType_;
    f32 radius_;
    Vector3 position_;
    Vector3 normal_;
    Vector3 emission_;
    Vector3 color_;
    MaterialType materialType_;

    Shape(Type shapeType, f32 radius, const Vector3& position, const Vector3& normal, const Vector3& emission, const Vector3& color, MaterialType materialType)
        :shapeType_(shapeType)
        ,radius_(radius)
        ,position_(position)
        ,normal_(normal)
        ,emission_(emission)
        ,color_(color)
        ,materialType_(materialType)
    {}

    // returns distance, (std::numeric_limits<f32>::max)() if nohit
    f32 intersect(const Ray& ray) const
    {
        switch(shapeType_){
        case Type_Sphere:
            return intersectSphere(ray);
        case Type_Plane:
            return intersectPlane(ray);
        }
    }

    void calcIntersection(Intersection& intersection, const Ray& ray) const
    {
        intersection.position_ = ray.origin_ + intersection.t_ * ray.direction_;

        switch(shapeType_){
        case Type_Sphere:
            intersection.normal_ = normalizeChecked(intersection.position_ - position_, Vector3::Zero);
            break;
        case Type_Plane:
            intersection.normal_ = normal_;
            break;
        }

        orthonormalBasis(intersection.binormal0_, intersection.binormal1_, intersection.normal_);
    }

    f32 intersectSphere(const Ray& ray) const;
    f32 intersectPlane(const Ray& ray) const;
};

f32 Shape::intersectSphere(const Ray& ray) const
{
    Vector3 m = ray.origin_ - position_;

    f32 b = dot(m, ray.direction_);
    f32 c = dot(m, m) - radius_ * radius_;

    if(0.0f < c) {
        if(0.0f < b) {
            return (std::numeric_limits<f32>::max)();
        }
    } else {
        return F32_EPSILON;
    }

    f32 discr = b * b - c;
    if(discr < LRENDER_RAY_EPSILON) {
        return (std::numeric_limits<f32>::max)();
    }

    discr = std::sqrtf(discr);
    b = -b;

    f32 tmin = b - discr;
    f32 tmax = b + discr;
    return 0.0f <= tmin ? tmin : LRENDER_RAY_EPSILON < tmax ? tmax : (std::numeric_limits<f32>::max)();
}

f32 Shape::intersectPlane(const Ray& ray) const
{
    f32 t = radius_ - dot(normal_, ray.origin_);
    f32 d = dot(normal_, ray.direction_);
    t /= d;

    return (d<F32_EPSILON && 0.0f <= t && t < (std::numeric_limits<f32>::max)()) ? t : (std::numeric_limits<f32>::max)();
}

std::array<Shape, 9> shapes{//Scene: type, radius, position, normal, emission, color, material
    Shape(Shape::Type_Plane, -5.5e-1f, Vector3::Zero, Vector3(-1.0f, 0.0f, 0.0f), Vector3::Zero,Vector3(.75f, .25f, .25f),DIFFUSE),//Left
    Shape(Shape::Type_Plane, -5.5e-1f, Vector3::Zero, Vector3( 1.0f, 0.0f, 0.0f), Vector3::Zero,Vector3(.25f, .25f, .75f),DIFFUSE),//Rght
    Shape(Shape::Type_Plane, -4e-1f, Vector3::Zero, Vector3(0.0f, 0.0f, -1.0f), Vector3::Zero,Vector3(.25f, .25f, .25f), DIFFUSE),//Back
    Shape(Shape::Type_Plane, -4e-1f, Vector3::Zero, Vector3(0.0f, 0.0f,  1.0f),  Vector3::Zero,Vector3(.75f, .75f, .25f), DIFFUSE),//Frnt
    Shape(Shape::Type_Plane, -0e-1f, Vector3::Zero, Vector3(0.0f,  1.0f, 0.0f), Vector3::Zero,Vector3(.75f, .75f, .75f),DIFFUSE),//Botm
    Shape(Shape::Type_Plane, -80e-2f, Vector3::Zero, Vector3(0.0f, -1.0f, 0.0f),Vector3::Zero,Vector3(.75f, .75f, .75f),DIFFUSE),//Top

    Shape(Shape::Type_Sphere, 20e-2f, Vector3(-30.0e-2f, 20e-2f, -10e-2f), Vector3::Zero, Vector3::Zero,Vector3(1,1,1)*.999f, ROUGH_CONDUCTOR),//Mirr
    Shape(Shape::Type_Sphere, 20e-2f, Vector3(25e-2f, 20e-2f, 1e-1f), Vector3::Zero, Vector3::Zero,Vector3(1,1,1)*.999f, DIFFUSE),//Glas
    Shape(Shape::Type_Sphere, 100e-2f, Vector3(0.0f,17.9e-1f, 0.0f), Vector3::Zero, Vector3(2,2,2), Vector3::Zero, EMITTER), //Light
};

bool intersect(Intersection& intersection, const Ray& ray)
{
    intersection.t_ = (std::numeric_limits<f32>::max)();
    intersection.id_ = -1;
    for(s32 i=0; i<shapes.size(); ++i){
        f32 d = shapes[i].intersect(ray);
        if(d<intersection.t_){
            intersection.t_ = d;
            intersection.id_ = i;
        }
    }
    if(0<=intersection.id_){
        shapes[intersection.id_].calcIntersection(intersection, ray);
        return true;
    }
    return false;
}

class EmitterConstant
{
public:
    static Vector3 eval(const Intersection& intersection, const Vector3& d, const Vector3& radiance);
};

Vector3 EmitterConstant::eval(const Intersection& intersection, const Vector3& d, const Vector3& radiance)
{
    return (F32_EPSILON < dot(intersection.normal_, d))? radiance : Vector3::Zero;
}

template<class T>
Vector3 radiance(Ray ray, s32 maxDepth, f32 roughness, lrender::Xoshiro128Plus& random)
{
    BSDF<T> bsdf(roughness, roughness, 1.0f, 1.8f, 0.0f);
    Vector3 Li = Vector3::Zero;
    Vector3 throughput = Vector3::One;
    f32 pdf = 1.0f;
    Intersection intersection;
    for(s32 depth = 0; depth<maxDepth; ++depth){
        if(!intersect(intersection, ray)){
            break;
        }
        const Shape& shape = shapes[intersection.id_];

        Vector3 wow = -ray.direction_;
        Vector3 wo = intersection.worldToLocal(wow);
        Vector3 n = intersection.worldToLocal(intersection.normal_);
        Vector3 wiw;

        f32 bsdfPdf = 0.0f;
        switch(shape.materialType_) {
        case DIFFUSE: {
            Diffuse diffuse;
            f32 eta0 = random.frand2();
            f32 eta1 = random.frand2();
            Vector3 wi;
            Vector3 bsdfWeight = diffuse.sample(wi, bsdfPdf, wo, eta0, eta1);
            Vector3 f = shape.color_ * bsdfWeight;
            if(f.isEqual(Vector3::Zero)) {
                return Li;
            }
            wiw = intersection.localToWorld(wi);
            throughput *= f;
        } break;
        case ROUGH_CONDUCTOR:{
            f32 eta0 = random.frand2();
            f32 eta1 = random.frand2();
            Vector3 wi;
            Vector3 bsdfWeight = bsdf.sample(wi, bsdfPdf, wo, eta0, eta1);
            Vector3 f = shape.color_ * bsdfWeight;
            if(f.isEqual(Vector3::Zero)) {
                return Li;
            }
            wiw = intersection.localToWorld(wi);
            throughput *= f;
        }break;
        case EMITTER:
            bsdfPdf = 1.0f;
            break;
        default:
            return Li;
        }

        if(!shape.emission_.isEqual(Vector3::Zero)){
            Li += throughput * bsdfPdf * EmitterConstant::eval(intersection, -ray.direction_, shape.emission_);
        }

        ray.origin_= intersection.position_;//muladd(LRENDER_RAY_EPSILON, intersection.normal_, intersection.position_);
        ray.direction_ = wiw;

        //Russian roulette
        if(6 <= depth) {
            f32 continueProbability = minimum(throughput.length(), 0.9f);
            if(continueProbability <= random.frand2()) {
                break;
            }
            throughput /= continueProbability;
        }
    }
    return Li;
}

/**
Schretter Colas, Kobbelt Leif, Dehaye Paul-Olivier, "Golden Ratio Sequences for Low-Discrepancy Sampling", JCGT 2012
*/
void golden_set(Vector2* points, s32 numSamples, lrender::Xoshiro128Plus& random)
{
    static const f32 magic = 0.618033988749894;
    f32 x = random.frand2();
    f32 min = x;
    s32 index = 0;
    for(s32 i=0; i<numSamples; ++i){
        points[i].y_ = x;

        if(x<min){
            min = x;
            index = i;
        }
        x += magic;
        if(1.0f<=x){
            x -= 1.0f;
        }
    }
    s32 f = 1;
    s32 fp = 1;
    s32 parity = 0;
    while((f+fp)<numSamples){
        s32 tmp = f;
        f += fp;
        fp = tmp;
        ++parity;
    }
    s32 inc, dec;
    if(parity&0x01U){
        inc = f;
        dec = fp;
    }else{
        inc = fp;
        dec = f;
    }
    points[0].x_ = points[index].y_;
    for(s32 i=1; i<numSamples; ++i){
        if(index<dec){
            index += inc;
            if(numSamples<=index){
                index -= dec;
            }
        }else{
            index -= dec;
        }
        points[i].x_ = points[index].y_;
    }
    f32 y = random.frand2();
    for(s32 i=0; i<numSamples; ++i){
        points[i].y_ = y;
        y += magic;
        if(1.0f<=y){
            y -= 1.0f;
        }
    }
}

template<class T>
void render(const char* name, s32 width, s32 height, s32 spp, u32 seed, f32 roughness)
{
    Vector2* samples = GFX_NEW Vector2[spp];
    lrender::Xoshiro128Plus random(seed);
    golden_set(samples, spp, random);

    static const s32 Width = width;
    static const s32 Height = height;

    f32 aspect = static_cast<f32>(Width)/Height;
    f32 angle = 65.0f*DEG_TO_RAD;
    f32 fovx = std::tanf(angle*0.5f);
    f32 fovy = fovx/aspect;

    Ray camera(Vector3(0.0f, 40.8e-2f, 120e-2f), normalize(Vector3(0, -0.042612f, -1)));
    Vector3 cx = Vector3(fovx, 0.0f, 0.0f);
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
                //f32 rx = random.frand2();
                //f32 ry = random.frand2();
                f32 rx = samples[s].x_;
                f32 ry = samples[s].y_;
                f32 sx = (x + rx)*invWidth2 - 1.0f;
                f32 sy = (y + ry)*invHeight2 - 1.0f;

                Vector3 d = normalize(cx*sx + cy*sy + camera.direction_);
                Ray ray;
                ray.origin_ = camera.origin_ + d*30e-4f;
                ray.direction_ = d;
                r += radiance<T>(ray, 16, roughness, random);
            }
            r *= invSpp;
            screen[row+x] += Vector3(clamp01(r.x_), clamp01(r.y_), clamp01(r.z_));
        }
    }

    timer.stop();
    GFX_DELETE_ARRAY(samples);
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

int main(int /*argc*/, char** /*argv*/)
{
    const s32 width = 1024;
    const s32 height = 768;
    const s32 samples = 4096;
    const u32 seed = 12345;
    const f32 roughness = 0.5f;

    //fresnel();
    //test<BeckmannIsotropic>(roughness, seed, "beckmann_out.csv");
    //test<GGXIsotropic>(roughness, seed, "ggx_out.csv");
    //test<GGXAnisotropicVND>(roughness, seed, "ggx_aniso_out.csv");
    //test<GGXAnisotropicElipsoidVND>(roughness, seed, "ggx_aniso_out2.csv");

    render<BeckmannIsotropic>("beckmann.ppm", width, height, samples, seed, roughness);
    render<GGXIsotropic>("ggx.ppm", width, height, samples, seed, roughness);
    render<GGXAnisotropicVND>("ggx_vnd.ppm", width, height, samples, seed, roughness);
    render<GGXAnisotropicElipsoidVND>("ggx_vnd2.ppm", width, height, samples, seed, roughness);
    return 0;
}
