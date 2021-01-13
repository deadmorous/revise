cmake_minimum_required (VERSION 3.1.3) 

get_filename_component(ProjectId ${CMAKE_CURRENT_SOURCE_DIR} NAME)
project(${ProjectId})

set(CMAKE_INCLUDE_CURRENT_DIR ON)

aux_source_directory(. SOURCE_FILES)

option(S3VS_WORKER_REPORT_DFIELD_TIME "Report time of dense field operations to stdout" OFF)
option(S3VS_WORKER_CUDA_RENDERING "Use CUDA for rendering, instead of OpenGL" ON)

if (S3VS_WORKER_CUDA_RENDERING)
    if (!S3DMM_ENABLE_CUDA)
        message(FATAL "If S3VS_WORKER_CUDA_RENDERING is enabled, S3DMM_ENABLE_CUDA must be enabled as well")
    endif()
    aux_source_directory(cuda RENDERING_SOURCE_FILES)
    add_definitions(-DS3VS_WORKER_CUDA_RENDERING -DS3VS_RENDERING_TYPE=OpenGL)
else (S3VS_WORKER_CUDA_RENDERING)
    find_package(VL COMPONENTS VLCore VLGraphics VLVG VLVolume VLMolecule REQUIRED)
    include_directories(${VL_INCLUDE_DIRS})
    set(OpenGL_GL_PREFERENCE GLVND)
    find_package(OpenGL REQUIRED)
    find_package(GLEW REQUIRED)
    aux_source_directory(opengl RENDERING_SOURCE_FILES)
    add_definitions(-DS3VS_RENDERING_TYPE=CUDA)
endif (S3VS_WORKER_CUDA_RENDERING)

find_package(SilverBulletsFactory)
include_directories(${SILVER_BULLETS_FACTORY_INCLUDE_DIRS})

find_package(Qt5 COMPONENTS Core Widgets REQUIRED)

# TODO: Remove
#aux_source_directory(opengl OTHER_TMP_SOURCE_FILES_TODO_REMOVE)

add_library(${PROJECT_NAME}
    ${SOURCE_FILES}
    ${RENDERING_SOURCE_FILES}
#    ${OTHER_TMP_SOURCE_FILES_TODO_REMOVE}
    )

add_dependencies(${PROJECT_NAME} s3dmm tecplot_rw)

target_link_libraries(${PROJECT_NAME}
    ${Boost_LIBRARIES}
    ${SILVER_BULLETS_FACTORY_LIBRARIES}
    s3dmm
    tecplot_rw
    Qt5::Core
    Qt5::Widgets
    stdc++fs
    )

if (NOT S3VS_WORKER_CUDA_RENDERING)
    target_link_libraries(${PROJECT_NAME}
        ${VL_LIBRARIES}
        ${OPENGL_LIBRARIES}
        ${GLEW_LIBRARIES}
        EGL
    #    GLESv2
        )
endif (NOT S3VS_WORKER_CUDA_RENDERING)

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

if (S3DMM_ENABLE_CUDA)
    add_dependencies(${PROJECT_NAME} s3dmm_cuda)
    target_link_libraries(${PROJECT_NAME} s3dmm_cuda)
endif()

if (S3VS_WORKER_REPORT_DFIELD_TIME)
    add_definitions(-DS3VS_WORKER_REPORT_DFIELD_TIME)
endif()

set_property(TARGET ${PROJECT_NAME} APPEND PROPERTY
  COMPILE_DEFINITIONS $<$<CONFIG:Debug>:_DEBUG>
)
