file(GLOB_RECURSE MATH_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

add_library(math STATIC ${MATH_SRC})

target_include_directories(math PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")