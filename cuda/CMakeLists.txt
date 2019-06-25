cmake_minimum_required(VERSION 3.12)

set(CMAKE_CONFIGURATION_TYPES "Debug" "Release")

set(ProjectName TestCudaBSDF)
project(${ProjectName})

find_package(CUDA)
message("Found CUDA: ${CUDA_FOUND} ${CUDA_VERSION}")

set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_PROPAGATE_HOST_FLAGS ON)

set(SOURCES "main.cpp;kernel.cu;kernel.h;BSDF.cu;BSDF.h;Coordinate.cu;Coordinate.h;lcuda.cu;lcuda.cpp;lcuda.h")
set(FILES ${SOURCES})

set(OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${OUTPUT_DIRECTORY}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${OUTPUT_DIRECTORY}")

cuda_add_executable(${ProjectName} ${FILES})

if(MSVC)
    set(DEFAULT_CXX_FLAGS "/DWIN32 /D_WINDOWS /D_MBCS /W4 /WX- /nologo /fp:precise /arch:AVX /Zc:wchar_t /TP /Gd")
    if("1800" VERSION_LESS MSVC_VERSION)
        set(DEFAULT_CXX_FLAGS "${DEFAULT_CXX_FLAGS} /EHsc")
    endif()

    set(CMAKE_CXX_FLAGS "${DEFAULT_CXX_FLAGS}")
    set(CMAKE_CXX_FLAGS_DEBUG "/D_DEBUG /MDd /Zi /Ob0 /Od /RTC1 /Gy /GR- /GS /Gm-")
    set(CMAKE_CXX_FLAGS_RELEASE "/MD /O2 /GL /GR- /DNDEBUG")

elseif(UNIX)
    set(DEFAULT_CXX_FLAGS "-Wall -O2 -std=c++14")
    set(CMAKE_CXX_FLAGS "${DEFAULT_CXX_FLAGS}")
elseif(APPLE)
endif()

set_target_properties(${ProjectName} PROPERTIES OUTPUT_NAME_DEBUG "${ProjectName}" OUTPUT_NAME_RELEASE "${ProjectName}")
set_property(TARGET ${ProjectName} PROPERTY CUDA_STANDARD 14)
