cmake_minimum_required (VERSION 3.1.3) 

get_filename_component(ProjectId ${CMAKE_CURRENT_SOURCE_DIR} NAME)
project(${ProjectId})

file(GLOB HEADER_FILES
    ${CMAKE_SOURCE_DIR}/include/${PROJECT_NAME}/*.hpp
    ${CMAKE_SOURCE_DIR}/include/${PROJECT_NAME}/*.h
)

include_directories(${S3DMM_ROOT_DIR}/dist/include)

aux_source_directory(. SOURCE_FILES)
add_library(${PROJECT_NAME} ${SOURCE_FILES} ${HEADER_FILES})

add_dependencies(${PROJECT_NAME} tecplot_rw)
target_link_libraries(${PROJECT_NAME} stdc++fs ${Boost_LIBRARIES} tecplot_rw)

set_property(TARGET ${PROJECT_NAME} APPEND PROPERTY
  COMPILE_DEFINITIONS $<$<CONFIG:Debug>:_DEBUG>
)

set_property(TARGET ${PROJECT_NAME} PROPERTY COMPILE_WARNING_AS_ERROR ON)
