cmake_minimum_required(VERSION 3.19)
project(test)

find_package(OpenCV REQUIRED)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(modules)

add_executable(test main.cpp)
target_include_directories(test
    PUBLIC
    "${PROJECT_SOURCE_DIR}/common/math/include"
    "${PROJECT_SOURCE_DIR}/modules/A1/include"
    "${PROJECT_SOURCE_DIR}/modules/A2/include"
    "${PROJECT_SOURCE_DIR}/modules/M1/include"
    "${PROJECT_SOURCE_DIR}/modules/M2/include"
    "${PROJECT_SOURCE_DIR}/common/kalman/include"
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(test PUBLIC A1 A2 M1 M2)
