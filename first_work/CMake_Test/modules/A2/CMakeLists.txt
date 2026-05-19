find_package(OpenCV REQUIRED)

file(GLOB_RECURSE A2_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

add_library(A2 STATIC ${A2_SRC})

target_include_directories(A2 PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

target_compile_options(A2
    PRIVATE
    -include cstddef
)