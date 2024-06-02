cmake_minimum_required (VERSION 3.1.3) 

get_filename_component(ProjectId ${CMAKE_CURRENT_SOURCE_DIR} NAME)
project(${ProjectId})

set(CMAKE_INCLUDE_CURRENT_DIR ON)

aux_source_directory(. SOURCE_FILES)

find_package(SilverBulletsFactory)
include_directories(${SILVER_BULLETS_FACTORY_INCLUDE_DIRS})

find_package(SilverBulletsSystem)
include_directories(${SILVER_BULLETS_SYSTEM_INCLUDE_DIRS})

# TODO: solve VL_CHECK problem and remove it.
find_package(VL COMPONENTS VLCore REQUIRED)

option(${ProjectId}_USE_OPENMP "Enable openmp for this library" OFF)

if (${ProjectId}_USE_OPENMP)
    find_package(OpenMP)
    if (OPENMP_FOUND)
        set(${ProjectId}_OPENMP_FLAGS OpenMP::OpenMP_CXX)
    endif()
endif()

if (S3DMM_ENABLE_FRAME_IMAGE_PARTS_SAVING)
    find_package(Qt6 COMPONENTS Core Gui REQUIRED)
endif()

add_library(${PROJECT_NAME} ${SOURCE_FILES})

add_dependencies(${PROJECT_NAME} s3dmm tecplot_rw)
target_link_libraries(${PROJECT_NAME}
    ${Boost_LIBRARIES}
    ${VL_LIBRARIES}
    ${SILVER_BULLETS_FACTORY_LIBRARIES}
    ${SILVER_BULLETS_SYSTEM_LIBRARIES}
    s3dmm
    ${${ProjectId}_OPENMP_FLAGS}
    )
if (S3DMM_ENABLE_FRAME_IMAGE_PARTS_SAVING)
    target_link_libraries(${PROJECT_NAME}
        Qt6::Core
        Qt6::Gui
        )
endif()
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

set_property(TARGET ${PROJECT_NAME} APPEND PROPERTY
  COMPILE_DEFINITIONS $<$<CONFIG:Debug>:_DEBUG>
)
