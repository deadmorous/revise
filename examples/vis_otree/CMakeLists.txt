cmake_minimum_required (VERSION 3.1.3) 

get_filename_component(ProjectId ${CMAKE_CURRENT_SOURCE_DIR} NAME)
project(${ProjectId})

set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(OpenGL_GL_PREFERENCE GLVND)

find_package(Qt5 COMPONENTS Core Widgets OpenGL REQUIRED)

aux_source_directory(. SOURCE_FILES)
file (GLOB_RECURSE HEADER_FILES RELATIVE ${PROJECT_SOURCE_DIR} *.h *.hpp *.hxx)
file (GLOB_RECURSE RESOURCES RELATIVE ${PROJECT_SOURCE_DIR} *.qrc)
# file (GLOB_RECURSE WINRC_FILES RELATIVE ${PROJECT_SOURCE_DIR} *.rc)
# qt5_create_translation (QM_FILE ${SOURCE_FILE_LIST} ${HEADER_FILES} ${TS_FILE})
# source_group ("Generated Files" FILES ${MOC_SOURCES} ${QRC_SOURCES} ${QM_FILE})
# source_group ("Translation Files" FILES ${TS_FILE})

find_package(VL COMPONENTS VLCore VLGraphics VLVG VLVolume VLMolecule VLQt5 REQUIRED)
include_directories(${VL_INCLUDE_DIRS})
# find ${OPENGL_LIBRARIES} to be used later
find_package(OpenGL REQUIRED)

macro(copy_resource resourceRootPath resourceRelPath)
    configure_file(${resourceRootPath}/${resourceRelPath} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/data/${resourceRelPath} COPYONLY)
endmacro()

macro(copy_vl_resource resourcePath)
    copy_resource(${CMAKE_SOURCE_DIR}/third_parties/VisualizationLibrary/data ${resourcePath})
endmacro()

copy_vl_resource(glsl/volume_luminance_light.vs)
copy_vl_resource(glsl/volume_raycast_isosurface.fs)
copy_vl_resource(glsl/volume_raycast_isosurface_transp.fs)
copy_vl_resource(glsl/volume_raycast_mip.fs)
copy_vl_resource(glsl/volume_raycast01.fs)
copy_vl_resource(glsl/volume_raycast02.fs)
copy_vl_resource(glsl/volume_raycast03.fs)
copy_vl_resource(font/bitstream-vera/VeraMono.ttf)

copy_resource(${PROJECT_SOURCE_DIR}/data glsl/volume_raycast_isosurface_alpha.fs)
copy_resource(${PROJECT_SOURCE_DIR}/data glsl/volume_raycast_isosurface_transp_alpha.fs)

add_executable(${PROJECT_NAME}
        ${SOURCE_FILES}
#        ${TS_FILE}
#        ${QM_FILE}
#        ${WINRC_FILES}
        )
#if (UNIX)
#    set_target_properties (${PROJECT_NAME} PROPERTIES
#        COMPILE_FLAGS -fPIE
#        )
#endif(UNIX)

add_dependencies(${PROJECT_NAME} s3dmm tecplot_rw)
target_link_libraries(${PROJECT_NAME}
    ${Boost_LIBRARIES}
    ${VL_LIBRARIES}
    ${OPENGL_LIBRARIES}
    s3dmm
    tecplot_rw
    Qt5::Core
    Qt5::Widgets
    Qt5::OpenGL)

set_property(TARGET ${PROJECT_NAME} APPEND PROPERTY
  COMPILE_DEFINITIONS $<$<CONFIG:Debug>:_DEBUG>
)
