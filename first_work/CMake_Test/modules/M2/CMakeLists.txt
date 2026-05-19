find_package(OpenCV REQUIRED)

aux_source_directory(src Path_M2)

add_library(M2 STATIC ${Path_M2})

target_include_directories(M2
    PUBLIC
    "${PROJECT_SOURCE_DIR}/modules/A1/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${PROJECT_SOURCE_DIR}/common/kalman/include"
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(M2 PUBLIC A2 ${OpenCV_LIBS}
)