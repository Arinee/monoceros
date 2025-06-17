
set(TARGETS_LIBS framework utils gflags gtest gtest_main)
set(TARGETS_INCS ../../src)

file(GLOB_RECURSE ALL_TEST_SRCS *_test.cc)

foreach(CC_SRCS ${ALL_TEST_SRCS})
  get_filename_component(CC_TARGET ${CC_SRCS} NAME_WE)
  list(APPEND TARGETS_TESTS ${CC_TARGET})
  set(${CC_TARGET}.SRCS ${CC_SRCS})
endforeach()

include(../../cmake/option.cmake)
include(../../cmake/frame.cmake)
