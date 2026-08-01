cmake_minimum_required(VERSION 3.18)
project(pixelboost_sr LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

include(FetchContent)

# -----------------------------------------------------------------------------
# 1. Download stb image header library
# -----------------------------------------------------------------------------
FetchContent_Declare(
    stb
    GIT_REPOSITORY https://github.com/nothings/stb.git
    GIT_TAG        master
)
FetchContent_MakeAvailable(stb)

# -----------------------------------------------------------------------------
# 2. Find or Auto-Download ONNX Runtime C SDK
# -----------------------------------------------------------------------------
# Try finding ONNX Runtime on system or via ONNXRUNTIME_ROOT hint
find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_c_api.h
    HINTS ${ONNXRUNTIME_ROOT}/include /usr/local/include /usr/include
)

find_library(ONNXRUNTIME_LIBRARY onnxruntime
    HINTS ${ONNXRUNTIME_ROOT}/lib ${ONNXRUNTIME_ROOT}/lib64 /usr/local/lib /usr/lib
)

# If not found locally, automatically download official release binaries
if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
    message(STATUS "ONNX Runtime not found locally. Auto-downloading pre-built C SDK...")

    set(ORT_VERSION "1.18.0") # Stable C API release

    if(WIN32)
        set(ORT_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-win-x64-${ORT_VERSION}.zip")
    elseif(APPLE)
        set(ORT_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-osx-universal2-${ORT_VERSION}.tgz")
    else()
        set(ORT_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-${ORT_VERSION}.tgz")
    endif()

    FetchContent_Declare(
        onnxruntime_sdk
        URL ${ORT_URL}
    )
    FetchContent_MakeAvailable(onnxruntime_sdk)

    set(ONNXRUNTIME_INCLUDE_DIR ${onnxruntime_sdk_SOURCE_DIR}/include)

    if(WIN32)
        set(ONNXRUNTIME_LIBRARY ${onnxruntime_sdk_SOURCE_DIR}/lib/onnxruntime.lib)
        set(ONNXRUNTIME_DLL ${onnxruntime_sdk_SOURCE_DIR}/lib/onnxruntime.dll)
    else()
        file(GLOB ONNXRUNTIME_LIBRARY "${onnxruntime_sdk_SOURCE_DIR}/lib/libonnxruntime*")
    endif()
endif()

message(STATUS "ONNX Runtime Include: ${ONNXRUNTIME_INCLUDE_DIR}")
message(STATUS "ONNX Runtime Library: ${ONNXRUNTIME_LIBRARY}")

# -----------------------------------------------------------------------------
# 3. Native PixelBoost C Library
# -----------------------------------------------------------------------------
add_library(pixelboost STATIC native/pixelboost.c)

target_include_directories(pixelboost
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${ONNXRUNTIME_INCLUDE_DIR}
)

target_link_libraries(pixelboost PRIVATE ${ONNXRUNTIME_LIBRARY})

# -----------------------------------------------------------------------------
# 4. C++ Test Executable
# -----------------------------------------------------------------------------
add_executable(test_pixelboost test_main.cpp)

target_include_directories(test_pixelboost
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${stb_SOURCE_DIR}
        ${ONNXRUNTIME_INCLUDE_DIR}
)

target_link_libraries(test_pixelboost PRIVATE pixelboost ${ONNXRUNTIME_LIBRARY})

# Copy Windows .dll next to the executable so it runs without path issues
if(WIN32 AND ONNXRUNTIME_DLL)
    add_custom_command(TARGET test_pixelboost POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ONNXRUNTIME_DLL}"
        $<TARGET_FILE_DIR:test_pixelboost>
    )
endif()
