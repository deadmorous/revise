cmake_minimum_required (VERSION 3.1.3) 

get_filename_component(ProjectId ${CMAKE_CURRENT_SOURCE_DIR} NAME)
project(${ProjectId})

set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

find_package(Qt6 COMPONENTS Core WebSockets Gui REQUIRED)

# include_directories(${S3DMM_ROOT_DIR}/dist/include)

aux_source_directory(. SOURCE_FILES)
file (GLOB_RECURSE HEADER_FILES RELATIVE ${PROJECT_SOURCE_DIR} *.h *.hpp *.hxx)
file (GLOB_RECURSE RESOURCES RELATIVE ${PROJECT_SOURCE_DIR} *.qrc)
# file (GLOB_RECURSE WINRC_FILES RELATIVE ${PROJECT_SOURCE_DIR} *.rc)
# qt5_create_translation (QM_FILE ${SOURCE_FILE_LIST} ${HEADER_FILES} ${TS_FILE})
# source_group ("Generated Files" FILES ${MOC_SOURCES} ${QRC_SOURCES} ${QM_FILE})
# source_group ("Translation Files" FILES ${TS_FILE})

add_library(${PROJECT_NAME}
    ${SOURCE_FILES}
    ${HEADER_FILES}
    ${RESOURCES}
    #${TS_FILE}
    #${QM_FILE}
)

target_link_libraries(${PROJECT_NAME}
    ${Boost_LIBRARIES}
    Qt6::Core
    Qt6::WebSockets
    Qt6::Gui
)

set_property(TARGET ${PROJECT_NAME} APPEND PROPERTY
  COMPILE_DEFINITIONS $<$<CONFIG:Debug>:_DEBUG>
)

set_property(TARGET ${PROJECT_NAME} PROPERTY COMPILE_WARNING_AS_ERROR ON)
