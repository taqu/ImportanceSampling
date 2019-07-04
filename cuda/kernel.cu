/**
@file kernel.cu
@author t-sakai
@date 2019/06/24
*/
#include "kernel.h"
#include "BSDF.h"

using namespace lcuda;

LCUDA_DEVICE f32 Shape::intersect(const Ray& ray) const
{
    switch(shapeType_){
    case Type_Sphere:
        return intersectSphere(ray);
    case Type_Plane:
        return intersectPlane(ray);
    }
    return F32_MAX;
}

LCUDA_DEVICE void Shape::calcIntersection(Intersection& intersection, const Ray& ray) const
{
    intersection.position_ = ray.origin_ + intersection.t_ * ray.direction_;

    switch(shapeType_){
    case Type_Sphere:
        intersection.normal_ = normalizeChecked(intersection.position_ - position_, make_float3(0.0f));
        break;
    case Type_Plane:
        intersection.normal_ = normal_;
        break;
    }

    orthonormalBasis(intersection.binormal0_, intersection.binormal1_, intersection.normal_);
}

LCUDA_DEVICE f32 Shape::intersectSphere(const Ray& ray) const
{
    float3 m = ray.origin_ - position_;

    f32 b = dot(m, ray.direction_);
    f32 c = dot(m, m) - radius_ * radius_;

    if(0.0f < c) {
        if(0.0f < b) {
            return F32_MAX;
        }
    } else {
        return F32_EPSILON;
    }

    f32 discr = b * b - c;
    if(discr < RAY_EPSILON) {
        return F32_MAX;
    }

    discr = std::sqrtf(discr);
    b = -b;

    f32 tmin = b - discr;
    f32 tmax = b + discr;
    return 0.0f <= tmin ? tmin : RAY_EPSILON < tmax ? tmax : F32_MAX;
}

LCUDA_DEVICE f32 Shape::intersectPlane(const Ray& ray) const
{
    f32 t = radius_ - dot(normal_, ray.origin_);
    f32 d = dot(normal_, ray.direction_);
    t /= d;

    return (d<F32_EPSILON && 0.0f <= t && t < F32_MAX) ? t : F32_MAX;
}

LCUDA_DEVICE float3 EmitterConstant::eval(const Intersection& intersection, const float3& d, const float3& radiance)
{
    return (F32_EPSILON < dot(intersection.normal_, d))? radiance : make_float3(0.0f);
}

LCUDA_CONSTANT ConstantsRender g_constansRender;

namespace
{
    LCUDA_DEVICE bool intersect(Intersection& intersection, const Ray& ray, s32 numShapes, const Shape* shapes)
    {
        intersection.t_ = F32_MAX;
        intersection.id_ = -1;
        for(s32 i = 0; i<numShapes; ++i){
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

    template<class T>
    LCUDA_DEVICE float3 radiance(Ray ray, s32 maxDepth, f32 roughness, uint4& random, s32 numShapes, const Shape* shapes)
    {
        BSDF<T> bsdf(roughness, roughness, 1.0f, 1.8f, 0.0f);
        float3 Li = make_float3(0.0f);
        float3 throughput = make_float3(1.0f);
        //f32 pdf = 1.0f;
        Intersection intersection;
        for(s32 depth = 0; depth<maxDepth; ++depth){
            if(!intersect(intersection, ray, numShapes, shapes)){
                break;
            }
            const Shape& shape = shapes[intersection.id_];

            float3 wow = -ray.direction_;
            float3 wo = intersection.worldToLocal(wow);
            float3 n = intersection.worldToLocal(intersection.normal_);
            float3 wiw;

            f32 bsdfPdf = 0.0f;
            switch(shape.materialType_) {
            case DIFFUSE: {
                Diffuse diffuse;
                f32 eta0 = xoshiro128plus_frand(random);
                f32 eta1 = xoshiro128plus_frand(random);
                float3 wi;
                float3 bsdfWeight = diffuse.sample(wi, bsdfPdf, wo, eta0, eta1);
                float3 f = shape.color_ * bsdfWeight;
                if(isZero(f)) {
                    return Li;
                }
                wiw = intersection.localToWorld(wi);
                throughput *= f;
            } break;
            case ROUGH_CONDUCTOR:{
                f32 eta0 = xoshiro128plus_frand(random);
                f32 eta1 = xoshiro128plus_frand(random);
                float3 wi;
                float3 bsdfWeight = bsdf.sample(wi, bsdfPdf, wo, eta0, eta1);
                float3 f = shape.color_ * bsdfWeight;
                if(isZero(f)) {
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

            if(!isZero(shape.emission_)){
                Li += throughput * bsdfPdf * EmitterConstant::eval(intersection, -ray.direction_, shape.emission_);
            }

            ray.origin_ = intersection.position_;//muladd(LRENDER_RAY_EPSILON, intersection.normal_, intersection.position_);
            ray.direction_ = wiw;

            //Russian roulette
            if(6 <= depth) {
                f32 continueProbability = fminf(length(throughput), 0.9f);
                if(continueProbability <= xoshiro128plus_frand(random)) {
                    break;
                }
                throughput /= continueProbability;
            }
        }
        return Li;
    }

    template<class T>
    LCUDA_GLOBAL void render(float3* screen)
    {
        s32 x = blockIdx.x * blockDim.x + threadIdx.x;
        s32 y = blockIdx.y * blockDim.y + threadIdx.y;

        const s32 Width = g_constansRender.width_;
        const s32 Height = g_constansRender.height_;

        if(Width<=x || Height<=y){
            return;
        }

        uint4 random = xoshiro128plus_srand(g_constansRender.random_ + Width*y + x);

        f32 fovx = g_constansRender.fovx_;
        f32 fovy = g_constansRender.fovy_;
        s32 samplesPerStep = g_constansRender.samplesPerStep_;

        const Ray& camera = g_constansRender.cameraRay_;
        float3 cx = make_float3(fovx, 0.0f, 0.0f);
        float3 cy = normalize(cross(cx, camera.direction_)) * fovy;

        f32 invWidth2 = 2.0f/Width;
        f32 invHeight2 = 2.0f/Height;

        s32 row = (Height-y-1)*Width;
        float3 r = make_float3(0.0f);
        for(s32 s = 0; s<samplesPerStep; ++s){
            //f32 rx = xoshiro128plus_frand(random);
            //f32 ry = xoshiro128plus_frand(random);
            f32 rx = g_constansRender.samples_[s].x;
            f32 ry = g_constansRender.samples_[s].y;
            f32 sx = (x + rx)*invWidth2 - 1.0f;
            f32 sy = (y + ry)*invHeight2 - 1.0f;

            float3 d = normalize(cx*sx + cy*sy + camera.direction_);
            Ray ray;
            ray.origin_ = camera.origin_ + d*30e-4f;
            ray.direction_ = d;
            r += radiance<T>(ray, 16, g_constansRender.roughness_, random, ConstantsRender::NumShapes, g_constansRender.shapes_);
        }
        f32 count = g_constansRender.count_;
        f32 inv = 1.0f/(count + samplesPerStep);
        screen[row+x] = screen[row+x] * count*inv + r*inv;
    }

    LCUDA_GLOBAL void test_random(int N, float* result, uint4 random)
    {
        for(int i=0; i<N; ++i){
            result[i] = xoshiro128plus_frand(random);
        }
    }
}

void kernel_render(float3* screen, int blockSize, const ConstantsRender& constants, DistributionType distribution)
{
    cudaMemcpyToSymbol(g_constansRender, &constants, sizeof(ConstantsRender));
    dim3 dimBlock(blockSize, blockSize);
    dim3 dimGrid((constants.width_ + blockSize - 1)/blockSize, (constants.height_ + blockSize - 1)/blockSize);

    switch(distribution){
    case Distribution_Beckmann:
        render<BeckmannIsotropic><<<dimGrid, dimBlock>>>(screen);
        break;
    case Distribution_GGX:
        render<GGXIsotropic><<<dimGrid, dimBlock>>>(screen);
        break;
    case Distribution_GGXVND:
        render<GGXAnisotropicVND><<<dimGrid, dimBlock>>>(screen);
        break;
    case Distribution_GGXVND2:
        render<GGXAnisotropicElipsoidVND><<<dimGrid, dimBlock>>>(screen);
        break;
    }
}

void kernel_random(int N, float* result, int blockSize, uint4 random)
{
    test_random<<<1, blockSize>>>(N, result, random);
}
