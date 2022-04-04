cmake_minimum_required(VERSION 2.8.3)

project(robo_lib_test)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(catkin REQUIRED
  roscpp
  roslib
  robo_lib
  plot_lib
)
 
catkin_package(
  INCLUDE_DIRS 
    include
  CATKIN_DEPENDS 
    roscpp 
)

include_directories(
  include
  /usr/include/eigen3
  ${catkin_INCLUDE_DIRS}
)

# =====================================================
# =====================================================

add_executable(cart_space_trajectory src/cart_space_trajectory.cpp)

add_dependencies(cart_space_trajectory
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)

target_link_libraries(cart_space_trajectory
  ${catkin_LIBRARIES}
  armadillo
  robo_lib
  plot_lib
)


# =====================================================
# =====================================================

add_executable(kinematic_chain_test 
  src/kinematic_chain_test.cpp
)

target_link_libraries(kinematic_chain_test
  ${catkin_LIBRARIES}
  armadillo
  robo_lib
)
