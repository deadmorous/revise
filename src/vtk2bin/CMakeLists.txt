cmake_minimum_required (VERSION 3.1.3) 

get_filename_component(ProjectId ${CMAKE_CURRENT_SOURCE_DIR} NAME)
project(${ProjectId})

include_directories(${S3DMM_ROOT_DIR}/dist/include)

if(S3DMM_ENABLE_VTK)
    FIND_PACKAGE(VTK REQUIRED)
endif()

if (VTK_FOUND)
    include(${VTK_USE_FILE})
endif()

aux_source_directory(. SOURCE_FILES)
add_executable(${PROJECT_NAME} ${SOURCE_FILES})

target_link_libraries(${PROJECT_NAME}
    ${Boost_LIBRARIES}
    s3dmm)

if (VTK_FOUND)
    target_link_libraries(${PROJECT_NAME} ${VTK_LIBRARIES})
endif()

set_property(TARGET ${PROJECT_NAME} APPEND PROPERTY
  COMPILE_DEFINITIONS $<$<CONFIG:Debug>:_DEBUG>
)

set_property(TARGET ${PROJECT_NAME} PROPERTY COMPILE_WARNING_AS_ERROR ON)
