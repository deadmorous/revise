cmake_minimum_required (VERSION 3.1.3) 

get_filename_component(ProjectId ${CMAKE_CURRENT_SOURCE_DIR} NAME)
project(${ProjectId})

include_directories(${S3DMM_ROOT_DIR}/dist/include)

aux_source_directory(. SOURCE_FILES)

if (S3DMM_ENABLE_COLESO)
    find_package(ColESo)
endif()

if (ColESo_FOUND)
    add_definitions(-DENABLE_COLESO REQUIRED)
endif()

find_package(SilverBulletsFactory REQUIRED)

if (S3DMM_ENABLE_VTK)
    FIND_PACKAGE(VTK REQUIRED)
endif()

if (VTK_FOUND)
    include(${VTK_USE_FILE})
endif()

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
add_dependencies(${PROJECT_NAME} s3dmm tecplot_rw)
target_link_libraries(
    ${PROJECT_NAME}
    ${Boost_LIBRARIES}
    s3dmm
    tecplot_rw
    ${SILVER_BULLETS_FACTORY_LIBRARIES}
    pthread)

if (ColESo_FOUND)
    target_link_libraries(${PROJECT_NAME} ${COLESO_LIBRARIES})
endif()

if (VTK_FOUND)
    target_link_libraries(${PROJECT_NAME} ${VTK_LIBRARIES})
endif()

set_property(TARGET ${PROJECT_NAME} APPEND PROPERTY
  COMPILE_DEFINITIONS $<$<CONFIG:Debug>:_DEBUG>
)

if (S3DMM_ENABLE_CUDA)
    add_dependencies(${PROJECT_NAME} s3dmm_cuda)
    target_link_libraries(${PROJECT_NAME} s3dmm_cuda)
endif()

set_property(TARGET ${PROJECT_NAME} PROPERTY COMPILE_WARNING_AS_ERROR ON)
