cmake_minimum_required (VERSION 3.1.3)

get_filename_component(ProjectId ${CMAKE_CURRENT_SOURCE_DIR} NAME)
project(${ProjectId})

set(CMAKE_INCLUDE_CURRENT_DIR ON)

find_package(SilverBulletsFactory)
include_directories(${SILVER_BULLETS_FACTORY_INCLUDE_DIRS})

find_package(SilverBulletsSystem)
include_directories(${SILVER_BULLETS_SYSTEM_INCLUDE_DIRS})

aux_source_directory(. SOURCE_FILES)

include_directories(${S3DMM_ROOT_DIR}/src/s3vs_lib)

add_executable(${PROJECT_NAME}
        ${SOURCE_FILES}
        )

add_dependencies(${PROJECT_NAME} s3vs_lib)
target_link_libraries(${PROJECT_NAME}
    ${Boost_LIBRARIES}
    ${SILVER_BULLETS_FACTORY_LIBRARIES}
    ${SILVER_BULLETS_SYSTEM_LIBRARIES}
    stdc++fs
    s3vs_lib
    )
#-lGL -lEGL -lGLESv2


set_property(TARGET ${PROJECT_NAME} APPEND PROPERTY
  COMPILE_DEFINITIONS $<$<CONFIG:Debug>:_DEBUG>
)

set_property(TARGET ${PROJECT_NAME} PROPERTY COMPILE_WARNING_AS_ERROR ON)
