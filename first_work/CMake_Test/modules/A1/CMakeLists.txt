find_package(OpenCV REQUIRED)

file(GLOB_RECURSE A1_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

add_library(A1 STATIC ${A1_SRC})

target_include_directories(A1 PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")