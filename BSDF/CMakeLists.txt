cmake_minimum_required(VERSION 3.12)

set(CMAKE_CONFIGURATION_TYPES "Debug" "Release")

set(ProjectName TestBSDF)
project(${ProjectName})

set(SOURCES "main.cpp;BSDF.cpp;BSDF.h;Coordinate.cpp;Coordinate.h;Plane.cpp;Plane.h;Sphere.cpp;Sphere.h;LString.cpp;LString.h;Matrix.cpp;Matrix.h;Quaternion.cpp;Quaternion.h;Vector.cpp;Vector.h;core.cpp;core.h")

include_directories(AFTER ${CMAKE_CURRENT_SOURCE_DIR})

set(FILES ${SOURCES})

set(OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${OUTPUT_DIRECTORY}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${OUTPUT_DIRECTORY}")

add_library(WINMMLIB MODULE IMPORTED)
set_target_properties(WINMMLIB PROPERTIES IMPORTED_LOCATION "winmm.lib")


add_executable(${ProjectName} ${FILES})

if(MSVC)
    set(DEFAULT_CXX_FLAGS "/DWIN32 /D_WINDOWS /D_MBCS /W4 /WX- /nologo /fp:precise /arch:AVX /Zc:wchar_t /TP /Gd")
    if("1800" VERSION_LESS MSVC_VERSION)
        set(DEFAULT_CXX_FLAGS "${DEFAULT_CXX_FLAGS} /EHsc")
    endif()

    set(CMAKE_CXX_FLAGS "${DEFAULT_CXX_FLAGS}")
    set(CMAKE_CXX_FLAGS_DEBUG "/D_DEBUG /MTd /Zi /Ob0 /Od /RTC1 /Gy /GR- /GS /Gm-")
    set(CMAKE_CXX_FLAGS_RELEASE "/MT /O2 /GL /GR- /DNDEBUG")

    target_link_libraries(${ProjectName} WINMMLIB)

elseif(UNIX)
    set(DEFAULT_CXX_FLAGS "-Wall -O2 -std=c++11")
    set(CMAKE_CXX_FLAGS "${DEFAULT_CXX_FLAGS}")
elseif(APPLE)
endif()

set_target_properties(${ProjectName} PROPERTIES OUTPUT_NAME_DEBUG "${ProjectName}" OUTPUT_NAME_RELEASE "${ProjectName}")
