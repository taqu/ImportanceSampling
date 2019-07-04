#include <stdio.h>
#include "lcuda.h"
#include "kernel.h"
#include <array>
#include <random>

using namespace lcuda;


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

/**
Schretter Colas, Kobbelt Leif, Dehaye Paul-Olivier, "Golden Ratio Sequences for Low-Discrepancy Sampling", JCGT 2012
*/
void golden_set(float2* points, s32 numSamples, uint4& random)
{
    static const f32 magic = 0.618033988749894;
    f32 x = xoshiro128plus_frand(random);
    f32 min = x;
    s32 index = 0;
    for(s32 i=0; i<numSamples; ++i){
        points[i].y = x;

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
    points[0].x = points[index].y;
    for(s32 i=1; i<numSamples; ++i){
        if(index<dec){
            index += inc;
            if(numSamples<=index){
                index -= dec;
            }
        }else{
            index -= dec;
        }
        points[i].x = points[index].y;
    }
    f32 y = xoshiro128plus_frand(random);
    for(s32 i=0; i<numSamples; ++i){
        points[i].y = y;
        y += magic;
        if(1.0f<=y){
            y -= 1.0f;
        }
    }
}

int main(int argc, char** argv)
{
#if 1
    static const s32 Width = 1024;
    static const s32 Height = 768;
    static const s32 Size = Width*Height;
    static const s32 SamplesPerPixel = 4096;

    std::random_device device_random;

    float3* screen = lcuda::cudaMalloc<float3>(Size);
    float3* screenHost = lcuda::cudaMallocHost<float3>(Size);

    ConstantsRender constants;
    constants.width_ = Width;
    constants.height_ = Height;
    constants.aspect_ = static_cast<f32>(Width)/Height;
    constants.angle_ = DEG_TO_RAD*65.0f;
    constants.fovx_ = std::tanf(constants.angle_*0.5f);
    constants.fovy_ = constants.fovx_/constants.aspect_;

    constants.cameraRay_ = Ray(make_float3(0.0f, 40.8e-2f, 120e-2f), normalize(make_float3(0.0f, -0.042612f, -1.0f)));

    constants.roughness_ = 0.5f;
    s32 spp = (SamplesPerPixel + ConstantsRender::SamplesPerStep -1) & ~(ConstantsRender::SamplesPerStep -1);
    constants.samplesPerStep_ = ConstantsRender::SamplesPerStep;

    //Scene: type, radius, position, normal, emission, color, material
    float3 Zero = make_float3(0.0f);
    constants.shapes_[0] = Shape(Shape::Type_Plane, -5.5e-1f, Zero, make_float3(-1.0f, 0.0f, 0.0f), Zero,make_float3(.75f, .25f, .25f),DIFFUSE);//Left
    constants.shapes_[1] = Shape(Shape::Type_Plane, -5.5e-1f, Zero, make_float3( 1.0f, 0.0f, 0.0f), Zero,make_float3(.25f, .25f, .75f),DIFFUSE);//Rght
    constants.shapes_[2] = Shape(Shape::Type_Plane, -4e-1f, Zero, make_float3(0.0f, 0.0f, -1.0f), Zero,make_float3(.25f, .25f, .25f), DIFFUSE);//Back
    constants.shapes_[3] = Shape(Shape::Type_Plane, -4e-1f, Zero, make_float3(0.0f, 0.0f,  1.0f),  Zero,make_float3(.75f, .75f, .25f), DIFFUSE);//Frnt
    constants.shapes_[4] = Shape(Shape::Type_Plane, -0e-1f, Zero, make_float3(0.0f,  1.0f, 0.0f), Zero,make_float3(.75f, .75f, .75f),DIFFUSE);//Botm
    constants.shapes_[5] = Shape(Shape::Type_Plane, -80e-2f, Zero, make_float3(0.0f, -1.0f, 0.0f),Zero,make_float3(.75f, .75f, .75f),DIFFUSE);//Top

    constants.shapes_[6] = Shape(Shape::Type_Sphere, 20e-2f, make_float3(-30.0e-2f, 20e-2f, -10e-2f), Zero, Zero,make_float3(1,1,1)*.999f, ROUGH_CONDUCTOR);//Mirr
    constants.shapes_[7] = Shape(Shape::Type_Sphere, 20e-2f, make_float3(25e-2f, 20e-2f, 1e-1f), Zero, Zero,make_float3(1,1,1)*.999f, DIFFUSE);//Glas
    constants.shapes_[8] = Shape(Shape::Type_Sphere, 100e-2f, make_float3(0.0f,17.9e-1f, 0.0f), Zero, make_float3(2,2,2), Zero, EMITTER); //Light

    s32 numLoops = spp/ConstantsRender::SamplesPerStep;

    std::array<DistributionType, 4> distributions = {Distribution_Beckmann, Distribution_GGX, Distribution_GGXVND, Distribution_GGXVND2};
    std::array<const char*, 4> filenames = {"beckmann.ppm", "ggx.ppm", "ggx_vnd.ppm", "ggx_vnd2.ppm"};

    for(s32 i = 0; i<distributions.size(); ++i){
        cudaMemset(screen, 0, sizeof(float3)*Size);
        uint4 random = xoshiro128plus_srand(device_random());
        constants.count_ = 0;
        for(s32 j = 0; j<numLoops; ++j){
            golden_set(constants.samples_, ConstantsRender::SamplesPerStep, random);
            constants.random_ = xoshiro128plus_rand(random);
            kernel_render(screen, 16, constants, distributions[i]);
            constants.count_ += ConstantsRender::SamplesPerStep;
        }

        cudaMemcpy(screenHost, screen, sizeof(float3)*Size, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        FILE* f = fopen(filenames[i], "w"); // Write image to PPM file.
        if(LCUDA_NULL == f){
            continue;
        }
        fprintf(f, "P3\n%d %d\n%d\n", Width, Height, 255);
        for(s32 j = 0; j<(Width*Height); ++j){
            float3 c = screenHost[j];
            u8 r = toU8(linearToSRGB(c.x));
            u8 g = toU8(linearToSRGB(c.y));
            u8 b = toU8(linearToSRGB(c.z));
            fprintf(f, "%d %d %d ", r, g, b);
        }
        fclose(f);
    }

    lcuda::cudaFreeHost(screenHost);
    lcuda::cudaFree(screen);
#else
    static const s32 Size = 1024;

    float* result = lcuda::cudaMalloc<float>(Size);
    float* resultHost = lcuda::cudaMallocHost<float>(Size);

    kernel_random(Size, result, 64, xoshiro128plus_srand(12345));
    cudaMemcpy(resultHost, result, sizeof(float)*Size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    FILE *f = fopen("random.csv", "w"); // Write image to PPM file.
    for(s32 i = 0; i<Size; ++i){
        fprintf(f, "%f\n", resultHost[i]);
    }
    fclose(f);
    lcuda::cudaFreeHost(resultHost);
    lcuda::cudaFree(result);
#endif

    return 0;
}
