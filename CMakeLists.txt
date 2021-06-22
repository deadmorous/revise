cmake_minimum_required (VERSION 3.1.3) 
project (s3dmm_package)

set (CMAKE_CXX_STANDARD 17)

set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
set (BUILD_SHARED_LIBS true)

set (S3DMM_ROOT_DIR ${CMAKE_CURRENT_LIST_DIR})

set (CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${S3DMM_ROOT_DIR}/dist)
set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${S3DMM_ROOT_DIR}/dist/cmake ${S3DMM_ROOT_DIR}/cmake)

include_directories(${PROJECT_SOURCE_DIR}/include)

option (S3DMM_PROGRESS_REPORT "Enable progress reports" ON)
set(S3DMM_REAL_TYPE float CACHE STRING "Real type (double or float)")

add_definitions (-DS3DMM_REAL_TYPE=${S3DMM_REAL_TYPE})

option(S3DMM_ENABLE_GUI "Enable GUI" ON)
option(S3DMM_ENABLE_REN "Enable offscreen renderer" ON)
option(S3DMM_BUILD_EXAMPLES "Build examples" ON)
option(S3DMM_ENABLE_CUDA "Enable CUDA" ON)
option(S3DMM_BUILD_S3VS_JS "Build s3vs_js (typically used only for development)" OFF)
option(S3DMM_BUILD_WS_SENDFRAME_JS "Build ws_sendframe_js (typically used only for development)" OFF)
option(S3DMM_ENABLE_COLESO "Enable ColESo" OFF)

option(S3DMM_ENABLE_WORKER_TIME_ESTIMATION "Enable worker time estimation and dumping" OFF)
option(S3DMM_ENABLE_FRAME_IMAGE_PARTS_SAVING "Enable image parts saving for each worker/assembly" OFF)

option(S3DMM_ENABLE_VTK "Enable code relying on the VTK library" OFF)
option(S3DMM_ENABLE_CGNS "Enable code relying on the CGNS library" OFF)

if (!S3DMM_ENABLE_GUI)
    message(FATAL "S3DMM_ENABLE_REN is enabled, but S3DMM_ENABLE_GUI is disabled. Please enable S3DMM_ENABLE_GUI")
endif()

if (S3DMM_PROGRESS_REPORT)
    list(APPEND REQUIRED_BOOST_LIBRARIES timer chrono system)
    add_definitions(-DS3DMM_PROGRESS_REPORT)
endif()

if (S3DMM_ENABLE_CUDA)
    add_definitions(-DS3DMM_ENABLE_CUDA)
endif()

if(S3DMM_ENABLE_WORKER_TIME_ESTIMATION)
    add_definitions(-DS3DMM_ENABLE_WORKER_TIME_ESTIMATION)
endif()

if(S3DMM_ENABLE_FRAME_IMAGE_PARTS_SAVING)
    add_definitions(-DS3DMM_ENABLE_FRAME_IMAGE_PARTS_SAVING)
endif()

if(S3DMM_ENABLE_VTK)
    add_definitions(-DS3DMM_ENABLE_VTK)
endif()

if(S3DMM_ENABLE_CGNS)
    add_definitions(-DS3DMM_ENABLE_CGNS)
endif()

list(APPEND REQUIRED_BOOST_LIBRARIES program_options filesystem)

# set (Boost_DEBUG ON)

if(WIN32)
    set(Boost_USE_STATIC_LIBS        ON)
    set(Boost_USE_STATIC_RUNTIME    OFF)
endif()
set(Boost_USE_MULTITHREADED      ON)
find_package(Boost REQUIRED ${REQUIRED_BOOST_LIBRARIES})
include_directories (${Boost_INCLUDE_DIRS})

add_subdirectory(src)
if (S3DMM_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Add tests
add_subdirectory (tests)

