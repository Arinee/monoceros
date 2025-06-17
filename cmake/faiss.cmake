cmake_minimum_required(VERSION 2.8)

include(ExternalProject)
ExternalProject_Add(
  faiss
  GIT_REPOSITORY http://gitlab+deploy-token-154:uraqr9ZcMu9hzzu_hFbb@gitlab.hutaojie.com/vector-galaxy/faiss.git
  GIT_TAG "hnsw"
  SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/faiss/faiss"
  BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/faiss/build"
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  TEST_COMMAND ""
)
