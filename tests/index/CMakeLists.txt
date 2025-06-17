##
##  Copyright (C) The Software Authors. All rights reserved.
##
##  \file     CMakeLists.txt
##  \author   Hechong.xyf
##  \date     May 2018
##  \version  1.0
##  \brief    Detail cmake build script
##

set(TARGETS_LIBS framework index_lib cluster gtest gtest_main)
set(TARGETS_INCS ../../src)

file(GLOB_RECURSE ALL_TEST_SRCS *_test.cc)

foreach(CC_SRCS ${ALL_TEST_SRCS})
  get_filename_component(CC_TARGET ${CC_SRCS} NAME_WE)
  list(APPEND TARGETS_TESTS ${CC_TARGET})
  set(${CC_TARGET}.SRCS ${CC_SRCS})
endforeach()

file(COPY testdata DESTINATION ${PROJECT_BINARY_DIR})

include(../../cmake/option.cmake)
include(../../cmake/frame.cmake)
