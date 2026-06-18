cmake_minimum_required(VERSION 3.10)
project(q3_solver)

set(CMAKE_CXX_STANDARD 17)

# 查找依赖
find_package(Eigen3 REQUIRED)
find_package(osqp REQUIRED)
find_package(OsqpEigen REQUIRED)

add_executable(q3_solver osqp_q3.cpp)

# 链接库
target_link_libraries(q3_solver
    Eigen3::Eigen
    osqp::osqp
    OsqpEigen::OsqpEigen
)
