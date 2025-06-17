##
##  Copyright (C) The Software Authors. All rights reserved.
##  Use of this source code is governed by a BSD-style
##  license that can be found in the LICENSE file.
##
##  \file     frame.cmake
##  \author   Rainvan(Yunfeng.Xiao)
##  \date     Nov 2013
##  \version  1.0
##  \brief    Detail cmake frame build script (C/C++)
##

## The following variables used by user's CMakeLists.txt:
##   TARGET_DIRS          (MUST)      e.g. dir1 dir2 dir3 ...
##   TARGET_OTHER_DIRS    (MUST)      e.g. other1 other2 other3 ...
##   TARGETS              (MUST)      e.g. libxxx.a libxxx.so xxx ...
##   TARGETS_TESTS        (MUST)      e.g. test1 test2 test3 ...
##   TARGETS_CFLAGS       (OPTION)    e.g. -fPIC ...
##   TARGETS_CXXFLAGS     (OPTION)    e.g. -fPIC ...
##   TARGETS_LDFLAGS      (OPTION)    e.g. -fPIC ...
##   TARGETS_INCS         (OPTION)    e.g. ./include ...
##   TARGETS_DEFS         (OPTION)    e.g. MACRO1 MACRO2 MACRO3=3 ...
##   TARGETS_LIBS         (OPTION)    e.g. -Ldir1 -llib1 ...
##   <TARGET>.TYPE        (OPTION)    e.g. SHARED, EXEC_SHARED, STATIC, EXEC
##   <TARGET>.ENTRY       (OPTION)    e.g. custom_main
##   <TARGET>.SRCS        (MUST)      e.g. *.c *.cc *.cpp ...
##   <TARGET>.INCS        (OPTION)    e.g. ./include ...
##   <TARGET>.PUBINCS     (OPTION)    e.g. ./include ...
##   <TARGET>.DEFS        (OPTION)    e.g. MACRO1 MACRO2 MACRO3=3 ...
##   <TARGET>.LIBS        (OPTION)    e.g. -Ldir1 -llib1 ...
##   <TARGET>.LDFLAGS     (OPTION)    e.g. -fPIC ...
##   <TARGET>.DEPS        (OPTION)    e.g. libxxx.so / dir1 ...
##   <TARGET>.ARGS        (OPTION)    e.g. arg1 arg2 ... (for tests)
##   <TARGET>.PROPERTIES  (OPTION)    e.g. WINDOWS_EXPORT_ALL_SYMBOLS ON

cmake_minimum_required(VERSION 2.8.12 FATAL_ERROR)

# Options of command
option(ENABLE_M32 "Enable 32-bit platform cross build" OFF)
option(ENABLE_M64 "Enable 64-bit platform cross build" OFF)
option(ENABLE_COVERAGE "Enable code coverage" OFF)

# Use rpaths on Mac OS X
set(CMAKE_MACOSX_RPATH ON)

# Export all symbol on Windows
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS OFF)

# Enable output of compile commands during generation
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Enable unit testing
enable_testing()

# unittest target
if(NOT TARGET unittest)
  add_custom_target(unittest COMMAND ${CMAKE_CTEST_COMMAND})
endif()

# Cross build flags
if(NOT MSVC)
  if(ENABLE_M64)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m64")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -m64")
  elseif(ENABLE_M32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m32")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -m32")
  endif()
else()
  # Microsoft VC compiler flags
  set(COMPILER_FLAGS
    CMAKE_CXX_FLAGS
    CMAKE_CXX_FLAGS_DEBUG
    CMAKE_CXX_FLAGS_RELEASE
    CMAKE_CXX_FLAGS_RELWITHDEBINFO
    CMAKE_CXX_FLAGS_MINSIZEREL
    CMAKE_C_FLAGS
    CMAKE_C_FLAGS_DEBUG
    CMAKE_C_FLAGS_RELEASE
    CMAKE_C_FLAGS_RELWITHDEBINFO
    CMAKE_C_FLAGS_MINSIZEREL
  )
  foreach(COMPILER_FLAG ${COMPILER_FLAGS})
    if(NOT BUILD_SHARED_LIBS)
      string(REPLACE "/MD" "/MT" ${COMPILER_FLAG} "${${COMPILER_FLAG}}")
    endif()
  endforeach()
endif()

# Directories of target output
if(NOT CMAKE_ARCHIVE_OUTPUT_DIRECTORY)
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
endif()
if(NOT CMAKE_LIBRARY_OUTPUT_DIRECTORY)
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
endif()
if(NOT CMAKE_RUNTIME_OUTPUT_DIRECTORY)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
endif()

set(__TEMP_DIRS ${TARGET_DIRS})
set(__TEMP_OTHER_DIRS ${TARGET_OTHER_DIRS})
unset(TARGET_DIRS)
unset(TARGET_OTHER_DIRS)

# Directories
foreach(TARGET_DIR ${__TEMP_DIRS})
  add_subdirectory(${TARGET_DIR} ${TARGET_DIR})
endforeach()

# Other Dependent Directories
foreach(TARGET_DIR ${__TEMP_OTHER_DIRS})
  add_subdirectory(${TARGET_DIR} ${TARGET_DIR} EXCLUDE_FROM_ALL)
endforeach()

set(TARGET_DIRS ${__TEMP_DIRS})
set(TARGET_OTHER_DIRS ${__TEMP_OTHER_DIRS})
unset(__TEMP_DIRS)
unset(__TEMP_OTHER_DIRS)

# Specific clang shared library compiling
if(${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
  set(CMAKE_SHARED_LIBRARY_CREATE_C_FLAGS 
      "${CMAKE_SHARED_LIBRARY_CREATE_C_FLAGS} -undefined dynamic_lookup")
  set(CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS 
      "${CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS} -undefined dynamic_lookup")
endif()

# Specific target defines
if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
  list(APPEND TARGETS_DEFS _GNU_SOURCE)
  set(LIBRARY_FLAGS
    CMAKE_SHARED_LIBRARY_CREATE_C_FLAGS
    CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS
  )
  foreach(LIBRARY_FLAG ${LIBRARY_FLAGS})
    string(REPLACE "-shared" "" ${LIBRARY_FLAG} "${${LIBRARY_FLAG}}")
  endforeach()
  set(TARGET_SHARED_LIBRARY_FLAGS "-shared")
  set(TARGET_EXEC_SHARED_LIBRARY_FLAGS "-pie -rdynamic")
endif()

# Compiler flags
if(NOT MSVC)
  include(CheckCCompilerFlag)
  include(CheckCXXCompilerFlag)
  CHECK_C_COMPILER_FLAG("-std=c99" COMPILER_SUPPORT_C99)
  CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORT_CXX11)
  CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORT_CXX0X)
  CHECK_CXX_COMPILER_FLAG("-std=c++14" COMPILER_SUPPORT_CXX14)
  if(COMPILER_SUPPORT_C99)
    list(APPEND TARGETS_CFLAGS -std=c99)
  endif()
  if(COMPILER_SUPPORT_CXX14)
    list(APPEND TARGETS_CXXFLAGS -std=c++14)
  elseif(COMPILER_SUPPORT_CXX11)
    list(APPEND TARGETS_CXXFLAGS -std=c++11)
  elseif(COMPILER_SUPPORT_CXX0X)
    list(APPEND TARGETS_CXXFLAGS -std=c++0x)
  endif()
  list(APPEND TARGETS_CFLAGS "-fPIC -Wall -Wextra -fdiagnostics-color=auto")
  list(APPEND TARGETS_CXXFLAGS "-fPIC -Wall -Wextra -fdiagnostics-color=auto")
else()
  # Microsoft VC compiler flags
  set(COMPILER_FLAGS
    CMAKE_CXX_FLAGS
    CMAKE_CXX_FLAGS_DEBUG
    CMAKE_CXX_FLAGS_RELEASE
    CMAKE_CXX_FLAGS_RELWITHDEBINFO
    CMAKE_CXX_FLAGS_MINSIZEREL
    CMAKE_C_FLAGS
    CMAKE_C_FLAGS_DEBUG
    CMAKE_C_FLAGS_RELEASE
    CMAKE_C_FLAGS_RELWITHDEBINFO
    CMAKE_C_FLAGS_MINSIZEREL
  )
  foreach(COMPILER_FLAG ${COMPILER_FLAGS})
    string(REPLACE "/W3" "/W4" ${COMPILER_FLAG} "${${COMPILER_FLAG}}")
  endforeach()
endif()

# C/C++ compiler flags
foreach(TARGETS_DEF ${TARGETS_DEFS})
  list(APPEND CMAKE_C_FLAGS -D${TARGETS_DEF})
  list(APPEND CMAKE_CXX_FLAGS -D${TARGETS_DEF})
endforeach()
set(CMAKE_C_FLAGS ${CMAKE_C_FLAGS} ${TARGETS_CFLAGS})
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${TARGETS_CXXFLAGS})
string(REPLACE ";" " " CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
string(REPLACE ";" " " CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

# C/C++ Linker flags
string(REPLACE ";" " " TARGETS_LDFLAGS "${TARGETS_LDFLAGS}")
set(CMAKE_EXE_LINKER_FLAGS ${TARGETS_LDFLAGS})
set(CMAKE_MODULE_LINKER_FLAGS ${TARGETS_LDFLAGS})
set(CMAKE_SHARED_LINKER_FLAGS ${TARGETS_LDFLAGS})

# Directories of header files
include_directories(${TARGETS_INCS})

# Targets
foreach(TARGET ${TARGETS})
  # Prepare
  unset(${TARGET}.COMPILE_FLAGS)
  unset(${TARGET}.LINK_FLAGS)
  unset(${TARGET}.LINK_LIBS)

  # Detect type of target
  if(NOT ${TARGET}.TYPE)
    set(__TEMP_TRIM ON)
    if((${TARGET} MATCHES "\\.so$") OR (${TARGET} MATCHES "\\.so\\.") OR 
      (${TARGET} MATCHES "\\.dll$") OR (${TARGET} MATCHES "\\.dylib$"))
      set(${TARGET}.TYPE "SHARED")
    elseif((${TARGET} MATCHES "\\.a$") OR (${TARGET} MATCHES "\\.lib$"))
      set(${TARGET}.TYPE "STATIC")
    else()
      set(${TARGET}.TYPE "EXEC")
      set(__TEMP_TRIM OFF)
    endif()
  else()
    set(__TEMP_TRIM OFF)
  endif()

  file(GLOB ${TARGET}.SRCS ${${TARGET}.SRCS})
  file(GLOB ${TARGET}.INCS ${${TARGET}.INCS})

  if(NOT ${TARGET}.SRCS)
    message(FATAL_ERROR "No source files found of target ${TARGET}.")
  endif()

  # Add target
  if(${${TARGET}.TYPE} STREQUAL "STATIC")
    add_library(${TARGET} STATIC ${${TARGET}.SRCS})
  elseif(${${TARGET}.TYPE} STREQUAL "SHARED")
    list(APPEND ${TARGET}.LINK_FLAGS ${TARGET_SHARED_LIBRARY_FLAGS})
    add_library(${TARGET} SHARED ${${TARGET}.SRCS})
  elseif(${${TARGET}.TYPE} STREQUAL "EXEC_SHARED")
    list(APPEND ${TARGET}.LINK_FLAGS ${TARGET_EXEC_SHARED_LIBRARY_FLAGS})
    add_library(${TARGET} SHARED ${${TARGET}.SRCS})
  elseif(${${TARGET}.TYPE} STREQUAL "EXEC")
    add_executable(${TARGET} ${${TARGET}.SRCS})
  elseif(${${TARGET}.TYPE} STREQUAL "PYTHON_MODULE")
    find_package(PythonLibs REQUIRED)
    if(PYTHON_LIBRARY)
      list(APPEND ${TARGET}.LIBS ${PYTHON_LIBRARY})
      list(APPEND ${TARGET}.INCS ${PYTHON_INCLUDE_DIR})
    endif()

    list(APPEND ${TARGET}.LINK_FLAGS ${TARGET_SHARED_LIBRARY_FLAGS})
    add_library(${TARGET} SHARED ${${TARGET}.SRCS})
    if(NOT MSVC)
      set_target_properties(${TARGET} PROPERTIES PREFIX "" SUFFIX ".so")
    else()
      set_target_properties(${TARGET} PROPERTIES PREFIX "" SUFFIX ".pyd" DEBUG_POSTFIX "_d")
    endif()
  else()
    message(FATAL_ERROR "Unknown Targeting Type: ${${TARGET}.TYPE}")
  endif()

  if(NOT MSVC)
    # Code coverage
    if(ENABLE_COVERAGE)
      list(APPEND ${TARGET}.COMPILE_FLAGS --coverage)
      list(APPEND ${TARGET}.LINK_FLAGS --coverage)
    endif()
    if(${TARGET}.ENTRY)
      list(APPEND ${TARGET}.LINK_FLAGS -nostartfiles -Wl,-e,${${TARGET}.ENTRY})
    endif()
  else()
    if(${TARGET}.ENTRY AND ({${TARGET}.TYPE} STREQUAL "EXEC"))
      list(APPEND ${TARGET}.LINK_FLAGS /entry:${${TARGET}.ENTRY})
    endif()
  endif()

  # Compiler flags
  foreach(TARGET_DEF ${${TARGET}.DEFS})
    list(APPEND ${TARGET}.COMPILE_FLAGS -D${TARGET_DEF})
  endforeach()

  # Linker flags
  list(APPEND ${TARGET}.LINK_FLAGS ${${TARGET}.LDFLAGS})

  # Linker libraries
  list(APPEND ${TARGET}.LINK_LIBS ${${TARGET}.LIBS} ${TARGETS_LIBS})

  string(REPLACE ";" " " ${TARGET}.COMPILE_FLAGS "${${TARGET}.COMPILE_FLAGS}")
  string(REPLACE ";" " " ${TARGET}.LINK_FLAGS "${${TARGET}.LINK_FLAGS}")

  # Public includes
  list(APPEND ${TARGET}.PUBLIC_INCS ${${TARGET}.PUBINCS})

  foreach(TARGET_DEP ${${TARGET}.DEPS})
    add_dependencies(${TARGET} ${TARGET_DEP})
    get_property(TARGET_DEP_INCS TARGET ${TARGET_DEP} PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
    message(STATUS "target:" ${TARGET})
    if(TARGET_DEP_INCS)
      list(APPEND ${TARGET}.PUBLIC_INCS ${TARGET_DEP_INCS})
    endif()
  endforeach()

  target_include_directories(${TARGET} PUBLIC ${${TARGET}.PUBLIC_INCS})
  target_include_directories(${TARGET} PRIVATE ${${TARGET}.INCS})
  target_link_libraries(${TARGET} ${${TARGET}.LINK_LIBS})
  if(__TEMP_TRIM)
    set_target_properties(${TARGET} PROPERTIES PREFIX "" SUFFIX "")
  endif()
  set_target_properties(${TARGET} PROPERTIES COMPILE_FLAGS "${${TARGET}.COMPILE_FLAGS}")
  set_target_properties(${TARGET} PROPERTIES LINK_FLAGS "${${TARGET}.LINK_FLAGS}")
  if(${TARGET}.PROPERTIES)
    set_target_properties(${TARGET} PROPERTIES ${${TARGET}.PROPERTIES})
  endif()
endforeach()

# Target Tests
foreach(TARGET ${TARGETS_TESTS})
  # Prepare
  unset(${TARGET}.COMPILE_FLAGS)
  unset(${TARGET}.LINK_FLAGS)
  unset(${TARGET}.LINK_LIBS)

  file(GLOB ${TARGET}.SRCS ${${TARGET}.SRCS})
  file(GLOB ${TARGET}.INCS ${${TARGET}.INCS})

  # Add target
  add_executable(${TARGET} EXCLUDE_FROM_ALL ${${TARGET}.SRCS})

  if(ENABLE_COVERAGE AND (NOT MSVC))
    set(${TARGET}.COMPILE_FLAGS  --coverage)
    set(${TARGET}.LINK_FLAGS  --coverage)
  endif()

  # Compiler flags
  if(NOT MSVC)
    list(APPEND ${TARGET}.COMPILE_FLAGS -g)
  endif()

  # Linker flags
  list(APPEND ${TARGET}.LINK_FLAGS ${${TARGET}.LDFLAGS})

  # Linker libraries
  list(APPEND ${TARGET}.LINK_LIBS ${${TARGET}.LIBS} ${TARGETS_LIBS})

  string(REPLACE ";" " " ${TARGET}.COMPILE_FLAGS "${${TARGET}.COMPILE_FLAGS}")
  string(REPLACE ";" " " ${TARGET}.LINK_FLAGS "${${TARGET}.LINK_FLAGS}")

  # Public includes
  list(APPEND ${TARGET}.PUBLIC_INCS ${${TARGET}.PUBINCS})

  foreach(TARGET_DEP ${${TARGET}.DEPS})
    add_dependencies(${TARGET} ${TARGET_DEP})
    get_property(TARGET_DEP_INCS TARGET ${TARGET_DEP} PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
    message(STATUS "target:" ${TARGET})
    if(TARGET_DEP_INCS)
      list(APPEND ${TARGET}.PUBLIC_INCS ${TARGET_DEP_INCS})
    endif()
  endforeach()

  target_include_directories(${TARGET} PUBLIC ${${TARGET}.PUBLIC_INCS})
  target_include_directories(${TARGET} PRIVATE ${${TARGET}.INCS})
  target_link_libraries(${TARGET} ${${TARGET}.LINK_LIBS})
  set_target_properties(${TARGET} PROPERTIES COMPILE_FLAGS "${${TARGET}.COMPILE_FLAGS}")
  set_target_properties(${TARGET} PROPERTIES LINK_FLAGS "${${TARGET}.LINK_FLAGS}")
  if(${TARGET}.PROPERTIES)
    set_target_properties(${TARGET} PROPERTIES ${${TARGET}.PROPERTIES})
  endif()
  add_test(NAME ${TARGET} 
    COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${TARGET}" ${${TARGET}.ARGS}
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
  add_custom_target(unittest.${TARGET}
    COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${TARGET}" ${${TARGET}.ARGS}
    DEPENDS ${TARGET}
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR})

  add_dependencies(unittest ${TARGET})
endforeach()
