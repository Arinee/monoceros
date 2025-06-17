
cmake_minimum_required(VERSION 2.8)

set(TARGET_DIRS framework utils cluster index builder service train)

if(ENABLE_FAISS)
    list(APPEND TARGET_DIRS faiss_index)
endif()

include(../cmake/frame.cmake)
