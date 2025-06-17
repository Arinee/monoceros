
cmake_minimum_required(VERSION 2.8)

include(ExternalProject)
ExternalProject_Add(
  thirdparty
  GIT_REPOSITORY http://gitlab+deploy-token-147:nUaUJ-UE2e1cXDjVisQk@gitlab.hutaojie.com/vector-galaxy/thirdparty.git
  GIT_TAG "master"
  SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/thirdparty/src"
  BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/thirdparty/build"
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  TEST_COMMAND ""
)
