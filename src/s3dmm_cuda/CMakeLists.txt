cmake_minimum_required (VERSION 3.18)

add_compile_options($<$<CONFIG:Debug>:-G>)

get_filename_component(ProjectId ${CMAKE_CURRENT_SOURCE_DIR} NAME)
project(${ProjectId} LANGUAGES CUDA)

option(S3DMM_CUDA_REPORT_DFIELD_TIME "Report time of dense field CUDA operations to stdout" OFF)

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
include_directories(${S3DMM_ROOT_DIR}/dist/include)

file(GLOB HEADER_FILES
    ${CMAKE_SOURCE_DIR}/include/${PROJECT_NAME}/*.hpp
    ${CMAKE_SOURCE_DIR}/include/${PROJECT_NAME}/*.h
)

file (GLOB_RECURSE SOURCE_FILES ${PROJECT_SOURCE_DIR} *.cu)

add_library(${PROJECT_NAME} ${SOURCE_FILES} ${HEADER_FILES})

target_link_libraries(${PROJECT_NAME} stdc++fs)

if (S3DMM_CUDA_REPORT_DFIELD_TIME)
    add_definitions(-DS3DMM_CUDA_REPORT_DFIELD_TIME)
endif()

set_property(TARGET ${PROJECT_NAME} APPEND PROPERTY
  COMPILE_DEFINITIONS $<$<CONFIG:Debug>:_DEBUG>
)

if (S3DMM_REAL_TYPE STREQUAL "double")
    set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES 61)
else()
    set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES 35 61)
endif()

set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_STANDARD 14)
