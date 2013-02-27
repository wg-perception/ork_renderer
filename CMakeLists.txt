cmake_minimum_required(VERSION 2.8.0)
project(object_recognition_renderer)

find_package(Boost REQUIRED)
find_package(OpenCV REQUIRED)

find_package(catkin REQUIRED)

option(USE_GLUT "Use GLUT instead of OSMesa" ON)

if (${USE_GLUT})
set(${PROJECT_NAME}_export_libs ${PROJECT_NAME}_glut)
set(${PROJECT_NAME}_include_files ${PROJECT_SOURCE_DIR}/include/${PROJECT_NAME}/renderer_glut.h)
else()
set(${PROJECT_NAME}_export_libs ${PROJECT_NAME}_osmesa)
set(${PROJECT_NAME}_include_files ${PROJECT_SOURCE_DIR}/include/${PROJECT_NAME}/renderer_osmesa.h)
endif()

if (${catkin_VERSION} VERSION_GREATER "0.5.28")
catkin_package(INCLUDE_DIRS include
               DEPENDS Boost
               LIBRARIES ${${PROJECT_NAME}_export_libs}
)
else()
catkin_stack()
catkin_project(object_recognition_renderer INCLUDE_DIRS include
                                           LIBRARIES ${${PROJECT_NAME}_export_libs}
)

set(CATKIN_PACKAGE_INCLUDE_DESTINATION include/${PROJECT_NAME})
endif()

include_directories(include)
include_directories(SYSTEM ${Boost_INCLUDE_DIRS}
                           ${catkin_INCLUDE_DIRS}
                           ${OpenCV_INCLUDE_DIRS}
)

add_subdirectory(src)

# install the include folder
install(FILES ${PROJECT_SOURCE_DIR}/include/${PROJECT_NAME}/renderer.h
                ${${PROJECT_NAME}_include_files}
        DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}/
)
