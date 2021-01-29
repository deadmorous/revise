cmake_minimum_required (VERSION 3.1.3) 

get_filename_component(ProjectId ${CMAKE_CURRENT_SOURCE_DIR} NAME)
project(${ProjectId})

set(CMAKE_INCLUDE_CURRENT_DIR ON)

find_package(Qt5 COMPONENTS Core Widgets REQUIRED)

include_directories(${S3DMM_ROOT_DIR}/dist/include)

aux_source_directory(. SOURCE_FILES)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

add_executable(${PROJECT_NAME}
        ${SOURCE_FILES}
        vis_timestamps.qrc
        )

target_link_libraries(${PROJECT_NAME}
    ${Boost_LIBRARIES}
    Qt5::Core
    Qt5::Widgets
    stdc++fs
    )

set_property(TARGET ${PROJECT_NAME} APPEND PROPERTY
  COMPILE_DEFINITIONS $<$<CONFIG:Debug>:_DEBUG>
)
