cmake_minimum_required (VERSION 3.1.3) 

get_filename_component(ProjectId ${CMAKE_CURRENT_SOURCE_DIR} NAME)
project(${ProjectId})

set(CMAKE_INCLUDE_CURRENT_DIR ON)

find_package(Qt6 COMPONENTS Core REQUIRED)

aux_source_directory(. SOURCE_FILES)
file (GLOB_RECURSE HEADER_FILES RELATIVE ${PROJECT_SOURCE_DIR} *.h *.hpp *.hxx)
file (GLOB_RECURSE RESOURCES RELATIVE ${PROJECT_SOURCE_DIR} *.qrc)

qt6_wrap_cpp (MOC_SOURCES ${HEADER_FILES})
qt6_add_resources (QRC_SOURCES ${RESOURCES})
source_group ("Generated Files" FILES ${MOC_SOURCES} ${QRC_SOURCES} ${QM_FILE})

add_executable(${PROJECT_NAME}
        ${SOURCE_FILES}
        ${HEADER_FILES}
        ${MOC_SOURCES}
        ${QRC_SOURCES}
        )

find_package(PNG REQUIRED)
include_directories(${PNG_INCLUDE_DIR})
target_link_libraries(${PROJECT_NAME} ${PNG_LIBRARY})

add_dependencies(${PROJECT_NAME} ws_sendframe)
target_link_libraries(${PROJECT_NAME}
    ${Boost_LIBRARIES}
    ws_sendframe
    Qt6::Core
    )

set_property(TARGET ${PROJECT_NAME} APPEND PROPERTY
  COMPILE_DEFINITIONS $<$<CONFIG:Debug>:_DEBUG>
)

set_property(TARGET ${PROJECT_NAME} PROPERTY COMPILE_WARNING_AS_ERROR ON)
