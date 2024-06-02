cmake_minimum_required (VERSION 3.1.3)

get_filename_component(ProjectId ${CMAKE_CURRENT_SOURCE_DIR} NAME)
project(${ProjectId})

set(CMAKE_INCLUDE_CURRENT_DIR ON)

find_package(Qt6 COMPONENTS Core Gui REQUIRED)

aux_source_directory(. SOURCE_FILES)

include_directories(${S3DMM_ROOT_DIR}/src/s3vs_lib)
find_package(VL COMPONENTS VLCore REQUIRED)
include_directories(${VL_INCLUDE_DIRS})

add_executable(${PROJECT_NAME}
        ${SOURCE_FILES}
        )

#add_dependencies(${PROJECT_NAME} s3vs_lib)
target_link_libraries(${PROJECT_NAME}
    ${Boost_LIBRARIES}
    Qt6::Core
    Qt6::Gui
    )
#-lGL -lEGL -lGLESv2


set_property(TARGET ${PROJECT_NAME} APPEND PROPERTY
  COMPILE_DEFINITIONS $<$<CONFIG:Debug>:_DEBUG>
)

set_property(TARGET ${PROJECT_NAME} PROPERTY COMPILE_WARNING_AS_ERROR ON)
