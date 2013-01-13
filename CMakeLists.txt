cmake_minimum_required(VERSION 2.8.3)
project(object_recognition_renderer)

find_package(catkin REQUIRED)

if (${catkin_VERSION} VERSION_GREATER "0.5.28")
catkin_package(INCLUDE_DIRS include
               LIBRARIES ${PROJECT_NAME}_osmesa
)
else()
catkin_stack()
catkin_project(object_recognition_renderer INCLUDE_DIRS include
                                           LIBRARIES ${PROJECT_NAME}_osmesa
)
endif()

include_directories(include)
include_directories(SYSTEM ${catkin_INCLUDE_DIRS})

add_subdirectory(src)

if (ROS_FUERTE_FOUND)
# install the include folder
install(PROGRAMS ${PROJECT_SOURCE_DIR}/include/
        DESTINATION include/
)
endif()
if (ROS_GROOVY_OR_ABOVE_FOUND)
# install the include folder
install(PROGRAMS ${PROJECT_SOURCE_DIR}/include/${PROJECT_NAME}
        DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)
endif()
