#ifndef INC_KERNEL_H_
#define INC_KERNEL_H_
/**
@file kernel.h
@author t-sakai
@date 2019/06/24
*/
#include "lcuda.h"

struct Intersection
{
    lcuda::s32 id_; //id of an intersected object
    lcuda::f32 t_; //distance to intersection
    float3 position_;
    float3 normal_;
    float3 binormal0_;
    float3 binormal1_;

    LCUDA_DEVICE float3 worldToLocal(const float3& v) const
    {
        return lcuda::normalize(make_float3(lcuda::dot(v, binormal0_), lcuda::dot(v, binormal1_), lcuda::dot(v, normal_)));
    }

    LCUDA_DEVICE float3 localToWorld(const float3& v) const
    {
        return make_float3(
            binormal0_.x * v.x + binormal1_.x * v.y + normal_.x * v.z,
            binormal0_.y * v.x + binormal1_.y * v.y + normal_.y * v.z,
            binormal0_.z * v.x + binormal1_.z * v.y + normal_.z * v.z);
    }
};

enum Type
{
    Type_Sphere,
    Type_Plane,
};

enum MaterialType
{
    EMITTER,
    DIFFUSE,
    ROUGH_CONDUCTOR,
};

struct Shape
{
    enum Type
    {
        Type_Sphere,
        Type_Plane,
    };

    Type shapeType_;
    lcuda::f32 radius_;
    float3 position_;
    float3 normal_;
    float3 emission_;
    float3 color_;
    MaterialType materialType_;

    LCUDA_HOST LCUDA_DEVICE Shape()
    {}

    LCUDA_HOST Shape(Type shapeType, lcuda::f32 radius, const float3& position, const float3& normal, const float3& emission, const float3& color, MaterialType materialType)
        :shapeType_(shapeType)
        ,radius_(radius)
        ,position_(position)
        ,normal_(normal)
        ,emission_(emission)
        ,color_(color)
        ,materialType_(materialType)
    {}

    // returns distance, (std::numeric_limits<f32>::max)() if nohit
    LCUDA_DEVICE lcuda::f32 intersect(const lcuda::Ray& ray) const;

    LCUDA_DEVICE void calcIntersection(Intersection& intersection, const lcuda::Ray& ray) const;

    LCUDA_DEVICE lcuda::f32 intersectSphere(const lcuda::Ray& ray) const;
    LCUDA_DEVICE lcuda::f32 intersectPlane(const lcuda::Ray& ray) const;
};

class EmitterConstant
{
public:
    LCUDA_DEVICE static float3 eval(const Intersection& intersection, const float3& d, const float3& radiance);
};

struct ConstantsRender
{
    static const lcuda::s32 NumShapes = 9;
    static const lcuda::s32 SamplesPerStep = 16;

    lcuda::s32 width_;
    lcuda::s32 height_;
    lcuda::f32 aspect_;
    lcuda::f32 angle_;
    lcuda::f32 fovx_;
    lcuda::f32 fovy_;

    lcuda::Ray cameraRay_;

    lcuda::f32 roughness_;
    lcuda::s32 spp_;
    lcuda::f32 scale_;
    lcuda::u32 random_;

    Shape shapes_[NumShapes];
    float2 samples_[SamplesPerStep];
};

enum DistributionType
{
    Distribution_Beckmann,
    Distribution_GGX,
    Distribution_GGXVND,
    Distribution_GGXVND2,
};

void kernel_render(float3* screen, int blockSize, const ConstantsRender& constants, DistributionType distribution);

void kernel_random(int N, float* result, int blockSize, uint4 random);
#endif //INC_KERNEL_H_
