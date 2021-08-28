# generated from genmsg/cmake/pkg-genmsg.cmake.em

message(STATUS "vio: 0 messages, 3 services")

set(MSG_I_FLAGS "-Istd_msgs:/opt/ros/melodic/share/std_msgs/cmake/../msg")

# Find all generators
find_package(gencpp REQUIRED)
find_package(geneus REQUIRED)
find_package(genlisp REQUIRED)
find_package(gennodejs REQUIRED)
find_package(genpy REQUIRED)

add_custom_target(vio_generate_messages ALL)

# verify that message/service dependencies have not changed since configure



get_filename_component(_filename "/root/Projects/ROS/src/p_33_vio/GPU_version/vio/srv/start.srv" NAME_WE)
add_custom_target(_vio_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "vio" "/root/Projects/ROS/src/p_33_vio/GPU_version/vio/srv/start.srv" ""
)

get_filename_component(_filename "/root/Projects/ROS/src/p_33_vio/GPU_version/vio/srv/getOdom.srv" NAME_WE)
add_custom_target(_vio_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "vio" "/root/Projects/ROS/src/p_33_vio/GPU_version/vio/srv/getOdom.srv" "std_msgs/Header"
)

get_filename_component(_filename "/root/Projects/ROS/src/p_33_vio/GPU_version/vio/srv/stop.srv" NAME_WE)
add_custom_target(_vio_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "vio" "/root/Projects/ROS/src/p_33_vio/GPU_version/vio/srv/stop.srv" ""
)

#
#  langs = gencpp;geneus;genlisp;gennodejs;genpy
#

### Section generating for lang: gencpp
### Generating Messages

### Generating Services
_generate_srv_cpp(vio
  "/root/Projects/ROS/src/p_33_vio/GPU_version/vio/srv/start.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/vio
)
_generate_srv_cpp(vio
  "/root/Projects/ROS/src/p_33_vio/GPU_version/vio/srv/getOdom.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/melodic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/vio
)
_generate_srv_cpp(vio
  "/root/Projects/ROS/src/p_33_vio/GPU_version/vio/srv/stop.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/vio
)

### Generating Module File
_generate_module_cpp(vio
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/vio
  "${ALL_GEN_OUTPUT_FILES_cpp}"
)

add_custom_target(vio_generate_messages_cpp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_cpp}
)
add_dependencies(vio_generate_messages vio_generate_messages_cpp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/root/Projects/ROS/src/p_33_vio/GPU_version/vio/srv/start.srv" NAME_WE)
add_dependencies(vio_generate_messages_cpp _vio_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/root/Projects/ROS/src/p_33_vio/GPU_version/vio/srv/getOdom.srv" NAME_WE)
add_dependencies(vio_generate_messages_cpp _vio_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/root/Projects/ROS/src/p_33_vio/GPU_version/vio/srv/stop.srv" NAME_WE)
add_dependencies(vio_generate_messages_cpp _vio_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(vio_gencpp)
add_dependencies(vio_gencpp vio_generate_messages_cpp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS vio_generate_messages_cpp)

### Section generating for lang: geneus
### Generating Messages

### Generating Services
_generate_srv_eus(vio
  "/root/Projects/ROS/src/p_33_vio/GPU_version/vio/srv/start.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/vio
)
_generate_srv_eus(vio
  "/root/Projects/ROS/src/p_33_vio/GPU_version/vio/srv/getOdom.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/melodic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/vio
)
_generate_srv_eus(vio
  "/root/Projects/ROS/src/p_33_vio/GPU_version/vio/srv/stop.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/vio
)

### Generating Module File
_generate_module_eus(vio
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/vio
  "${ALL_GEN_OUTPUT_FILES_eus}"
)

add_custom_target(vio_generate_messages_eus
  DEPENDS ${ALL_GEN_OUTPUT_FILES_eus}
)
add_dependencies(vio_generate_messages vio_generate_messages_eus)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/root/Projects/ROS/src/p_33_vio/GPU_version/vio/srv/start.srv" NAME_WE)
add_dependencies(vio_generate_messages_eus _vio_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/root/Projects/ROS/src/p_33_vio/GPU_version/vio/srv/getOdom.srv" NAME_WE)
add_dependencies(vio_generate_messages_eus _vio_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/root/Projects/ROS/src/p_33_vio/GPU_version/vio/srv/stop.srv" NAME_WE)
add_dependencies(vio_generate_messages_eus _vio_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(vio_geneus)
add_dependencies(vio_geneus vio_generate_messages_eus)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS vio_generate_messages_eus)

### Section generating for lang: genlisp
### Generating Messages

### Generating Services
_generate_srv_lisp(vio
  "/root/Projects/ROS/src/p_33_vio/GPU_version/vio/srv/start.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/vio
)
_generate_srv_lisp(vio
  "/root/Projects/ROS/src/p_33_vio/GPU_version/vio/srv/getOdom.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/melodic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/vio
)
_generate_srv_lisp(vio
  "/root/Projects/ROS/src/p_33_vio/GPU_version/vio/srv/stop.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/vio
)

### Generating Module File
_generate_module_lisp(vio
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/vio
  "${ALL_GEN_OUTPUT_FILES_lisp}"
)

add_custom_target(vio_generate_messages_lisp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_lisp}
)
add_dependencies(vio_generate_messages vio_generate_messages_lisp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/root/Projects/ROS/src/p_33_vio/GPU_version/vio/srv/start.srv" NAME_WE)
add_dependencies(vio_generate_messages_lisp _vio_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/root/Projects/ROS/src/p_33_vio/GPU_version/vio/srv/getOdom.srv" NAME_WE)
add_dependencies(vio_generate_messages_lisp _vio_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/root/Projects/ROS/src/p_33_vio/GPU_version/vio/srv/stop.srv" NAME_WE)
add_dependencies(vio_generate_messages_lisp _vio_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(vio_genlisp)
add_dependencies(vio_genlisp vio_generate_messages_lisp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS vio_generate_messages_lisp)

### Section generating for lang: gennodejs
### Generating Messages

### Generating Services
_generate_srv_nodejs(vio
  "/root/Projects/ROS/src/p_33_vio/GPU_version/vio/srv/start.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/vio
)
_generate_srv_nodejs(vio
  "/root/Projects/ROS/src/p_33_vio/GPU_version/vio/srv/getOdom.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/melodic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/vio
)
_generate_srv_nodejs(vio
  "/root/Projects/ROS/src/p_33_vio/GPU_version/vio/srv/stop.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/vio
)

### Generating Module File
_generate_module_nodejs(vio
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/vio
  "${ALL_GEN_OUTPUT_FILES_nodejs}"
)

add_custom_target(vio_generate_messages_nodejs
  DEPENDS ${ALL_GEN_OUTPUT_FILES_nodejs}
)
add_dependencies(vio_generate_messages vio_generate_messages_nodejs)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/root/Projects/ROS/src/p_33_vio/GPU_version/vio/srv/start.srv" NAME_WE)
add_dependencies(vio_generate_messages_nodejs _vio_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/root/Projects/ROS/src/p_33_vio/GPU_version/vio/srv/getOdom.srv" NAME_WE)
add_dependencies(vio_generate_messages_nodejs _vio_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/root/Projects/ROS/src/p_33_vio/GPU_version/vio/srv/stop.srv" NAME_WE)
add_dependencies(vio_generate_messages_nodejs _vio_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(vio_gennodejs)
add_dependencies(vio_gennodejs vio_generate_messages_nodejs)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS vio_generate_messages_nodejs)

### Section generating for lang: genpy
### Generating Messages

### Generating Services
_generate_srv_py(vio
  "/root/Projects/ROS/src/p_33_vio/GPU_version/vio/srv/start.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/vio
)
_generate_srv_py(vio
  "/root/Projects/ROS/src/p_33_vio/GPU_version/vio/srv/getOdom.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/melodic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/vio
)
_generate_srv_py(vio
  "/root/Projects/ROS/src/p_33_vio/GPU_version/vio/srv/stop.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/vio
)

### Generating Module File
_generate_module_py(vio
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/vio
  "${ALL_GEN_OUTPUT_FILES_py}"
)

add_custom_target(vio_generate_messages_py
  DEPENDS ${ALL_GEN_OUTPUT_FILES_py}
)
add_dependencies(vio_generate_messages vio_generate_messages_py)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/root/Projects/ROS/src/p_33_vio/GPU_version/vio/srv/start.srv" NAME_WE)
add_dependencies(vio_generate_messages_py _vio_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/root/Projects/ROS/src/p_33_vio/GPU_version/vio/srv/getOdom.srv" NAME_WE)
add_dependencies(vio_generate_messages_py _vio_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/root/Projects/ROS/src/p_33_vio/GPU_version/vio/srv/stop.srv" NAME_WE)
add_dependencies(vio_generate_messages_py _vio_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(vio_genpy)
add_dependencies(vio_genpy vio_generate_messages_py)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS vio_generate_messages_py)



if(gencpp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/vio)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/vio
    DESTINATION ${gencpp_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_cpp)
  add_dependencies(vio_generate_messages_cpp std_msgs_generate_messages_cpp)
endif()

if(geneus_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/vio)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/vio
    DESTINATION ${geneus_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_eus)
  add_dependencies(vio_generate_messages_eus std_msgs_generate_messages_eus)
endif()

if(genlisp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/vio)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/vio
    DESTINATION ${genlisp_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_lisp)
  add_dependencies(vio_generate_messages_lisp std_msgs_generate_messages_lisp)
endif()

if(gennodejs_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/vio)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/vio
    DESTINATION ${gennodejs_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_nodejs)
  add_dependencies(vio_generate_messages_nodejs std_msgs_generate_messages_nodejs)
endif()

if(genpy_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/vio)
  install(CODE "execute_process(COMMAND \"/usr/bin/python2\" -m compileall \"${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/vio\")")
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/vio
    DESTINATION ${genpy_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_py)
  add_dependencies(vio_generate_messages_py std_msgs_generate_messages_py)
endif()
