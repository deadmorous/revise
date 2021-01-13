cmake_minimum_required (VERSION 3.1.3) 

get_filename_component(ProjectId ${CMAKE_CURRENT_SOURCE_DIR} NAME)
project(${ProjectId})

set(CMAKE_INCLUDE_CURRENT_DIR ON)

find_package(Qt5 COMPONENTS Core Widgets REQUIRED)

aux_source_directory(. SOURCE_FILES)

add_executable(${PROJECT_NAME}
        ${SOURCE_FILES}
        )

add_dependencies(${PROJECT_NAME} s3dmm)
target_link_libraries(${PROJECT_NAME}
    ${Boost_LIBRARIES}
    s3dmm
    tecplot_rw
    Qt5::Core
    Qt5::Widgets
    )

set_property(TARGET ${PROJECT_NAME} APPEND PROPERTY
  COMPILE_DEFINITIONS $<$<CONFIG:Debug>:_DEBUG>
)
