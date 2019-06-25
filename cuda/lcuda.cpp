/**
@file lcuda.cpp
@author t-sakai
@date 2019/06/24
*/
#include "lcuda.h"

namespace lcuda
{
    void printError(cudaError error, const Char* file, s32 line)
    {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s)\n", file, line, (s32)error, cudaGetErrorName(error));
    }

    void printError(cudaError error, const Char* func, const Char* file, s32 line)
    {
        fprintf(stderr, "CUDA error at %s(%s:%d) code=%d(%s)\n", func, file, line, (s32)error, cudaGetErrorName(error));
    }

    /**
    From helper_cuda.h
    */
    s32 getSMVer2Cores(s32 major, s32 minor) {
        struct sSMtoCores
        {
            s32 version_; //0xMm, M=SM Major version, m=SM minor version
            s32 cores_;
        };

        sSMtoCores nGpuArchCoresPerSM[] =
        {
            {0x30, 192},
            {0x32, 192},
            {0x35, 192},
            {0x37, 192},
            {0x50, 128},
            {0x52, 128},
            {0x53, 128},
            {0x60, 64},
            {0x61, 128},
            {0x62, 128},
            {0x70, 64},
            {0x72, 64},
            {0x75, 64},
            {-1, -1}
        };

        s32 version = (major<<4) + minor;
        s32 i;
        for(i=0; 0<=nGpuArchCoresPerSM[i].version_; ++i){
            if(nGpuArchCoresPerSM[i].version_ == version){
                return nGpuArchCoresPerSM[i].cores_;
            }
        }
        return nGpuArchCoresPerSM[i-1].cores_;
    }

    s32 getMaximumPerformanceDevice()
    {
        s32 count = 0;
        CALL_RETURN_IF_FAILURE(cudaGetDeviceCount(&count), InvalidID);

        u64 maxComputePerformance = 0;
        cudaDeviceProp deviceProp;
        s32 maxPerformanceDevice = InvalidID;
        for(s32 deviceId=0; deviceId<count; ++deviceId){
            if(cudaSuccess != cudaGetDeviceProperties(&deviceProp, deviceId)){
                continue;
            }
            if(deviceProp.computeMode == cudaComputeModeProhibited){
                continue;
            }
            s32 smPerMultiprocessor;
            if(deviceProp.major == 9999 && deviceProp.minor == 9999){
                smPerMultiprocessor = 1;
            } else{
                smPerMultiprocessor = getSMVer2Cores(deviceProp.major, deviceProp.minor);
            }
            u64 computePerformance = (u64)deviceProp.multiProcessorCount * smPerMultiprocessor * deviceProp.clockRate;
            if(maxComputePerformance < computePerformance){
                maxComputePerformance = computePerformance;
                maxPerformanceDevice = deviceId;
            }
        }
        return maxPerformanceDevice;
    }

    s32 initializeDevice(s32 deviceId)
    {
        if(deviceId<0){
            deviceId = getMaximumPerformanceDevice();

        } else{
            s32 count = 0;
            CALL_RETURN_IF_FAILURE(cudaGetDeviceCount(&count), InvalidID);
            if(count <= 0 || count <= deviceId){
                return InvalidID;
            }
            cudaDeviceProp deviceProp;
            CALL_RETURN_IF_FAILURE(cudaGetDeviceProperties(&deviceProp, deviceId), InvalidID);
            if(cudaComputeModeProhibited == deviceProp.computeMode){
                return InvalidID;
            }
            if(deviceProp.major < 1){
                return InvalidID;
            }
        }

        if(0<=deviceId){
            CALL_RETURN_IF_FAILURE(cudaSetDevice(deviceId), InvalidID);
        }
        return deviceId;
    }
}
