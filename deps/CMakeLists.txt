
cmake_minimum_required(VERSION 2.8)
# Download and unpack thirdparty at configure time
configure_file(../cmake/thirdparty.cmake thirdparty/download/CMakeLists.txt)
execute_process(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/thirdparty/download")
execute_process(COMMAND "${CMAKE_COMMAND}" --build .
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/thirdparty/download")

add_subdirectory("${CMAKE_CURRENT_BINARY_DIR}/thirdparty/src"
                 "${CMAKE_CURRENT_BINARY_DIR}/thirdparty/build")

# Download and unpack faiss project at configure time
if(ENABLE_FAISS)
    configure_file(../cmake/faiss.cmake faiss/download/CMakeLists.txt)
    execute_process(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/faiss/download")
    execute_process(COMMAND "${CMAKE_COMMAND}" --build .
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/faiss/download")

    set(BUILD_TUTORIAL OFF CACHE BOOL "DO NOT Buildtutorials" FORCE)
    set(BUILD_TEST OFF CACHE BOOL "DO NOT Build tests" FORCE)
    set(BUILD_WITH_GPU OFF CACHE BOOL "DO NOT Build faiss with gpu" FORCE)
    set(WITH_MKL OFF CACHE BOOL "DO NOT Build with MKL" FORCE)
    # trick for faiss
    add_subdirectory("${CMAKE_CURRENT_BINARY_DIR}/faiss/faiss" 
        "${CMAKE_CURRENT_BINARY_DIR}/faiss/build" EXCLUDE_FROM_ALL)
endif()

