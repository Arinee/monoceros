
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD+Patents license found in the
# LICENSE file in the root directory of this source tree.

# -*- makefile -*-
# tested on CentOS 7, Ubuntu 16 and Ubuntu 14, see below to adjust flags to distribution.


CC=gcc
CXX=g++

CFLAGS=-fPIC -m64 -Wall -g -O3 -mavx -msse4 -mpopcnt -fopenmp -Wno-sign-compare -fopenmp
CXXFLAGS=$(CFLAGS) -std=c++11
LDFLAGS=-g -fPIC  -fopenmp

# common linux flags
SHAREDEXT=so
SHAREDFLAGS=-shared
FAISSSHAREDFLAGS=-shared


##########################################################################
# Uncomment one of the 4 BLAS/Lapack implementation options
# below. They are sorted # from fastest to slowest (in our
# experiments).
##########################################################################

#
# 1. Intel MKL
#
# This is the fastest BLAS implementation we tested. Unfortunately it
# is not open-source and determining the correct linking flags is a
# nightmare. See
#
#   https://software.intel.com/en-us/articles/intel-mkl-link-line-advisor
#
# The latest tested version is MLK 2017.0.098 (2017 Initial Release) and can
# be downloaded here:
#
#   https://registrationcenter.intel.com/en/forms/?productid=2558&licensetype=2
#
# The following settings are working if MLK is installed on its default folder:
#
# MKLROOT=/opt/intel/compilers_and_libraries/linux/mkl/
#
# BLASLDFLAGS=-Wl,--no-as-needed -L$(MKLROOT)/lib/intel64   -lmkl_intel_ilp64 \
# -lmkl_core -lmkl_gnu_thread -ldl -lpthread
#
# BLASCFLAGS=-DFINTEGER=long
#
# you may have to set the LD_LIBRARY_PATH=$MKLROOT/lib/intel64 at runtime.
# If at runtime you get the error:
#   Intel MKL FATAL ERROR: Cannot load libmkl_avx2.so or libmkl_def.so.
# You may add set
#   LD_PRELOAD=$MKLROOT/lib/intel64/libmkl_core.so:$MKLROOT/lib/intel64/libmkl_sequential.so
# at runtime as well.

#
# 2. Openblas
#
# The library contains both BLAS and Lapack. About 30% slower than MKL. Please see 
#   https://github.com/facebookresearch/faiss/wiki/Troubleshooting#slow-brute-force-search-with-openblas
# to fix performance problemes with OpenBLAS 

BLASCFLAGS=-DFINTEGER=int

# This is for Centos:
BLASLDFLAGS?=/usr/lib64/libopenblas.so.0

# for Ubuntu 16:
# sudo apt-get install libopenblas-dev python-numpy python-dev
# BLASLDFLAGS?=/usr/lib/libopenblas.so.0

# for Ubuntu 14:
# sudo apt-get install libopenblas-dev liblapack3 python-numpy python-dev
# BLASLDFLAGS?=/usr/lib/libopenblas.so.0 /usr/lib/lapack/liblapack.so.3.0
#
# 3. Atlas
#
# Automatically tuned linear algebra package. As the name indicates,
# it is tuned automatically for a give architecture, and in Linux
# distributions, it the architecture is typically indicated by the
# directory name, eg. atlas-sse3 = optimized for SSE3 architecture.
#
# BLASCFLAGS=-DFINTEGER=int
# BLASLDFLAGS=/usr/lib64/atlas-sse3/libptf77blas.so.3 /usr/lib64/atlas-sse3/liblapack.so
#
# 4. reference implementation
#
# This is just a compiled version of the reference BLAS
# implementation, that is not optimized at all.
#
# BLASCFLAGS=-DFINTEGER=int
# BLASLDFLAGS=/usr/lib64/libblas.so.3 /usr/lib64/liblapack.so.3.2
#


##########################################################################
# SWIG and Python flags
##########################################################################

# SWIG executable. This should be at least version 3.x
SWIGEXEC=swig

# The Python include directories for a given python executable can
# typically be found with
#
# python -c "import distutils.sysconfig; print distutils.sysconfig.get_python_inc()"
# python -c "import numpy ; print numpy.get_include()"
#
# or, for Python 3, with
#
# python3 -c "import distutils.sysconfig; print(distutils.sysconfig.get_python_inc())"
# python3 -c "import numpy ; print(numpy.get_include())"
#

PYTHONCFLAGS=-I/usr/include/python2.7/ -I/usr/lib64/python2.7/site-packages/numpy/core/include/


###########################################################################
# Cuda GPU flags
###########################################################################



# root of the cuda 8 installation
CUDAROOT=/usr/local/cuda-8.0/

CUDACFLAGS=-I$(CUDAROOT)/include -I../build/deps/aitheta/src/src -I../src

NVCC=$(CUDAROOT)/bin/nvcc

NVCCFLAGS= $(CUDAFLAGS) \
   -I $(CUDAROOT)/targets/x86_64-linux/include/ \
   -I ../build/deps/aitheta/src/src \
   -I ../src/ \
   -Xcompiler -fPIC \
   -Xcudafe --diag_suppress=unrecognized_attribute \
   -gencode arch=compute_35,code="compute_35" \
   -gencode arch=compute_52,code="compute_52" \
   -gencode arch=compute_60,code="compute_60" \
   --std c++11 -lineinfo \
   -ccbin $(CXX) -DFAISS_USE_FLOAT16

# BLAS LD flags for nvcc (used to generate an executable)
# if BLASLDFLAGS contains several flags, each one may
# need to be prepended with -Xlinker
BLASLDFLAGSNVCC=-Xlinker $(BLASLDFLAGS)

# Same, but to generate a .so
BLASLDFLAGSSONVCC=-Xlinker  $(BLASLDFLAGS)
