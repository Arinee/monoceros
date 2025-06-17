##
##  Copyright (C) The Software Authors. All rights reserved.
##
##  \file     CMakeLists.txt
##  \author   yunheng@xiaohongshu.com
##  \date     Jan 2019
##  \version  1.0
##  \brief    Detail cmake build script

cmake_minimum_required(VERSION 2.8)
project(mercury-project)

set(TARGET_DIRS deps src tests tools)
include(./cmake/frame.cmake)
