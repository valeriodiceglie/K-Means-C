cmake_minimum_required(VERSION 3.20)
project(kmeans-c)
add_compile_options( -ffast-math )

set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

add_compile_definitions(PB_C99_STATIC_ASSERT)
# Kmeans
set(KMEANS_PROJECT_DIR "${CMAKE_CURRENT_LIST_DIR}")
set(KMEANS_TEST_DIR "${KMEANS_PROJECT_DIR}/test")
set(KMEANS_SRC_DIR "${KMEANS_PROJECT_DIR}/src")

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/res
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

# -- Project settings
file(GLOB KMEANS_SOURCES CONFIGURE_DEPENDS
        ${KMEANS_SRC_DIR}/*.c)

file(GLOB KMEANS_TEST_SOURCES CONFIGURE_DEPENDS
        ${KMEANS_SRC_DIR}/*.c
        ${K_MEANS_DIR}/kmedoid.c
        ${KMEANS_TEST_DIR}/*.c
        ${KMEANS_TEST_DIR}/tests/*.c)

# -- Targets
add_library(kmeans ${KMEANS_SOURCES})
target_link_libraries(kmeans ${KMEANS_LIBS})
target_include_directories(kmeans PUBLIC ${KMEANS_TEST_HEADERS})


add_executable(kmeans_test ${KMEANS_TEST_SOURCES})
target_link_libraries(kmeans_test PRIVATE m ${KMEANS_LIBS})
target_include_directories(kmeans_test PUBLIC ${KMEANS_TEST_HEADERS})
# set_target_properties(kmeans_test PROPERTIES EXCLUDE_FROM_ALL True)
