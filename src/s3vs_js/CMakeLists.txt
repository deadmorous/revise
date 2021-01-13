cmake_minimum_required (VERSION 3.1.3) 

get_filename_component(ProjectId ${CMAKE_CURRENT_SOURCE_DIR} NAME)
project(${ProjectId})

set(CMAKE_INCLUDE_CURRENT_DIR ON)

aux_source_directory(src SOURCE_FILES)

find_path(NODE_API_INCLUDE_DIR node_api.h PATH_SUFFIXES node REQUIRED)

find_package(SilverBulletsFactory)
include_directories(${SILVER_BULLETS_FACTORY_INCLUDE_DIRS})

add_library(${PROJECT_NAME} ${SOURCE_FILES})

add_dependencies(${PROJECT_NAME} s3vs_lib)
target_link_libraries(${PROJECT_NAME}
    ${Boost_LIBRARIES}
    ${SILVER_BULLETS_FACTORY_LIBRARIES}
    s3vs_lib
    )

target_include_directories(${PROJECT_NAME} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/node_modules/node-addon-api
    ${NODE_API_INCLUDE_DIR})

set_property(TARGET ${PROJECT_NAME} APPEND PROPERTY
  COMPILE_DEFINITIONS $<$<CONFIG:Debug>:_DEBUG>
)
