cmake_minimum_required(VERSION 3.10)
project(rm_detect)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 14)

# 寻找 OpenCV 库
find_package(OpenCV REQUIRED)

# 包含 OpenCV 头文件
include_directories(${OpenCV_INCLUDE_DIRS})

# 将 main.cpp 编译成可执行文件
add_executable(rm_detect main.cpp)

# 链接 OpenCV 库
target_link_libraries(rm_detect ${OpenCV_LIBS})