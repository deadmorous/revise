cmake_minimum_required (VERSION 3.1.3) 

get_filename_component(ProjectId ${CMAKE_CURRENT_SOURCE_DIR} NAME)
project(${ProjectId})

set(CMAKE_INCLUDE_CURRENT_DIR ON)

find_package(Qt5 COMPONENTS Core Widgets Svg REQUIRED)

aux_source_directory(. SOURCE_FILES)
file (GLOB_RECURSE HEADER_FILES RELATIVE ${PROJECT_SOURCE_DIR} *.h *.hpp *.hxx)
file (GLOB_RECURSE RESOURCES RELATIVE ${PROJECT_SOURCE_DIR} *.qrc)

qt5_wrap_cpp (MOC_SOURCES ${HEADER_FILES})
qt5_add_resources (QRC_SOURCES ${RESOURCES})
# qt5_create_translation (QM_FILE ${SOURCE_FILE_LIST} ${HEADER_FILES} ${TS_FILE})
source_group ("Generated Files" FILES ${MOC_SOURCES} ${QRC_SOURCES} ${QM_FILE})
# source_group ("Translation Files" FILES ${TS_FILE})

add_executable(${PROJECT_NAME}
        ${SOURCE_FILES}
        ${HEADER_FILES}
        ${MOC_SOURCES}
        ${QRC_SOURCES}
        #${TS_FILE}
        #${QM_FILE}
        )

add_dependencies(${PROJECT_NAME} s3dmm)
target_link_libraries(${PROJECT_NAME}
    ${Boost_LIBRARIES}
    s3dmm
    tecplot_rw
    Qt5::Core
    Qt5::Widgets
    Qt5::Svg
    )

set_property(TARGET ${PROJECT_NAME} APPEND PROPERTY
  COMPILE_DEFINITIONS $<$<CONFIG:Debug>:_DEBUG>
)

set_property(TARGET ${PROJECT_NAME} PROPERTY COMPILE_WARNING_AS_ERROR ON)
