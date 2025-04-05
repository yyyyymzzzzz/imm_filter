cmake_minimum_required(VERSION 3.12)
project(imm_filter_test)

set(CMAKE_CXX_STANDARD 17)

find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Python COMPONENTS Interpreter Development)
include_directories(
    ${Python_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${CERES_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}
)

add_executable(imm_test
    main.cpp
    imm_filter.hpp
    adaptive_ekf.hpp
    filter.hpp
)

target_link_libraries(imm_test
    Eigen3::Eigen
    ${Python_LIBRARIES}
    ${CERES_LIBRARIES}
    ${OpenCV_LIBS}
)