cmake_minimum_required (VERSION 3.1.3) 

get_filename_component(ProjectId ${CMAKE_CURRENT_SOURCE_DIR} NAME)
project(${ProjectId})

set(CMAKE_INCLUDE_CURRENT_DIR ON)

find_package(Qt5 COMPONENTS Core Widgets REQUIRED)

find_package(SilverBulletsSystem)
include_directories(${SILVER_BULLETS_SYSTEM_INCLUDE_DIRS})

aux_source_directory(. SOURCE_FILES)

find_package(VL COMPONENTS VLCore VLGraphics VLVG VLVolume VLMolecule VLQt5 REQUIRED)
include_directories(${VL_INCLUDE_DIRS})
set(OpenGL_GL_PREFERENCE GLVND)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)

add_executable(${PROJECT_NAME}
    ${SOURCE_FILES}
)

#find_package(PNG REQUIRED)
#include_directories(${PNG_INCLUDE_DIR})
#target_link_libraries(${PROJECT_NAME} ${PNG_LIBRARY})

add_dependencies(${PROJECT_NAME} s3dmm tecplot_rw s3vs_worker)
target_link_libraries(${PROJECT_NAME}
    ${Boost_LIBRARIES}
    ${VL_LIBRARIES}
    ${OPENGL_LIBRARIES}
    ${GLEW_LIBRARIES}
    EGL
    ${SILVER_BULLETS_SYSTEM_LIBRARIES}
    s3dmm
    tecplot_rw
    s3vs_worker
    Qt5::Core
    Qt5::Widgets
)

set_property(TARGET ${PROJECT_NAME} APPEND PROPERTY
  COMPILE_DEFINITIONS $<$<CONFIG:Debug>:_DEBUG>
)
