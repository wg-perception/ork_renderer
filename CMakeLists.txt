cmake_minimum_required(VERSION 2.8.0)
project(object_recognition_renderer)

find_package(Boost REQUIRED)
find_package(OpenCV REQUIRED)

find_package(catkin REQUIRED)

option(USE_GLUT "Use GLUT instead of OSMesa" OFF)

if (${USE_GLUT})
find_package(GLUT)
set(${PROJECT_NAME}_depends GLUT)
endif()

catkin_package(INCLUDE_DIRS include
               DEPENDS Boost ${${PROJECT_NAME}_depends}
               LIBRARIES ${PROJECT_NAME}_2d ${PROJECT_NAME}_3d
)

include_directories(include)
include_directories(SYSTEM ${Boost_INCLUDE_DIRS}
                           ${catkin_INCLUDE_DIRS}
                           ${OpenCV_INCLUDE_DIRS}
)

add_subdirectory(src)

# install the include folder
install(FILES ${PROJECT_SOURCE_DIR}/include/${PROJECT_NAME}/renderer.h
              ${PROJECT_SOURCE_DIR}/include/${PROJECT_NAME}/renderer2d.h
              ${PROJECT_SOURCE_DIR}/include/${PROJECT_NAME}/renderer3d.h
              ${PROJECT_SOURCE_DIR}/include/${PROJECT_NAME}/utils.h
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}/
)

# add docs
if (COMMAND ecto_sphinx)
ecto_sphinx(${CMAKE_CURRENT_SOURCE_DIR}/doc/ doc)
endif()
