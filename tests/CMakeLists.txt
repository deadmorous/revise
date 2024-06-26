cmake_minimum_required (VERSION 3.1.3)
project (s3dmm_tests) 

set (GTEST_DIR ../third_parties/googletest/googletest)

include_directories(${GTEST_DIR}/include ${GTEST_DIR})

# aux_source_directory(. SOURCE_FILES)
file (GLOB_RECURSE SOURCE_FILES ./*.cpp)
add_executable(${PROJECT_NAME} ${GTEST_DIR}/src/gtest-all.cc ${GTEST_DIR}/src/gtest_main.cc ${SOURCE_FILES})

add_dependencies (
    ${PROJECT_NAME}
    s3dmm
    s3vs_worker
    )
target_link_libraries (
    ${PROJECT_NAME}
    s3dmm
    s3vs_worker)

if (UNIX)
    target_link_libraries (${PROJECT_NAME} pthread)
endif()

set_property(TARGET ${PROJECT_NAME} PROPERTY COMPILE_WARNING_AS_ERROR ON)
