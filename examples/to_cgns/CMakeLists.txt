cmake_minimum_required (VERSION 3.1.3) 

get_filename_component(ProjectId ${CMAKE_CURRENT_SOURCE_DIR} NAME)
project(${ProjectId})

set(CMAKE_INCLUDE_CURRENT_DIR ON)

find_package(SilverBulletsSystem)
find_package(CGNS REQUIRED)

include_directories(${SILVER_BULLETS_SYSTEM_INCLUDE_DIRS} ${CGNS_INCLUDE_DIRS})

aux_source_directory(. SOURCE_FILES)

add_executable(${PROJECT_NAME}
    ${SOURCE_FILES}
)

add_dependencies(${PROJECT_NAME} s3dmm tecplot_rw s3vs_worker)
target_link_libraries(${PROJECT_NAME}
    ${Boost_LIBRARIES}
    ${CGNS_LIBRARIES}
    ${SILVER_BULLETS_SYSTEM_LIBRARIES}
    s3dmm
    tecplot_rw
    /usr/lib/x86_64-linux-gnu/libhdf5_serial.so
)

set_property(TARGET ${PROJECT_NAME} APPEND PROPERTY
  COMPILE_DEFINITIONS $<$<CONFIG:Debug>:_DEBUG>
)
