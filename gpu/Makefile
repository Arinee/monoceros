# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD+Patents license found in the
# LICENSE file in the root directory of this source tree.

.SUFFIXES: .cpp .o .cu

include ./makefile.inc

LIBNAME=libgpuproxima

all: $(LIBNAME).a $(LIBNAME).$(SHAREDEXT) examples/test_knn_linear examples/test_knn_linear_ip
unittest: test/gpuknn_linear_test
#all: $(LIBNAME).a 

CPPOBJ=     gpu_knn_linear.o \
			utils/GpuResources.o  \
            utils/StandardGpuResources.o \
            utils/DeviceUtils.o \
            utils/StackDeviceMemory.o \
            utils/MemorySpace.o \
	    	utils/DeviceMemory.o 


CUOBJ=      impl/linear_index.o \
			impl/index_calculator.o \
			impl/index_distance.o

.cpp.o:
	$(CXX) $(CXXFLAGS) -fPIC -m64 -Wall -g -Wno-sign-compare -O3 -fopenmp \
         -c $< -o $@ $(EXTRAFLAGS) $(CUDACFLAGS)

.cu.o:
	$(NVCC) $(NVCCFLAGS) -g -O3 \
         -c $< -o $@ $(EXTRAFLAGS)


$(LIBNAME).a: $(CPPOBJ) $(CUOBJ)
	ar r $@ $^

$(LIBNAME).$(SHAREDEXT): $(CPPOBJ) $(CUOBJ)
	$(CXX) -L$(CUDAROOT)/lib64 -lcublas -lcudart $(LDFLAGS) \
	$(FAISSSHAREDFLAGS) -o $(LIBNAME).$(SHAREDEXT) $^


examples/test_knn_linear: examples/test_knn_linear.cpp \
	$(LIBNAME).a 
	$(NVCC) $(NVCCFLAGS) -o $@ $^ -Xcompiler -fopenmp -lcublas \
	$(BLASLDFLAGSNVCC)
examples/test_knn_linear_ip: examples/test_knn_linear_ip.cpp \
	$(LIBNAME).a 
	$(NVCC) $(NVCCFLAGS) -o $@ $^ -Xcompiler -fopenmp -lcublas \
	$(BLASLDFLAGSNVCC)
test/gpuknn_linear_test: test/gpuknn_linear_test.cc \
	$(LIBNAME).$(SHAREDEXT)  ../build/lib/libgtest.a ../build/lib/libgtest_main.a ../build/lib/libknn_linear.so ../build/lib/libaitheta.so
	$(NVCC) $(NVCCFLAGS) -o $@ $^ -Xcompiler -fopenmp -lcublas \
	$(BLASLDFLAGSNVCC) -I ../build/deps/thirdparty/src/googletest-1.8.0/googletest/include

clean:
	rm -rf *.o impl/*.o utils/*.o test/*.o examples/*.o $(LIBNAME).a \
		 $(LIBNAME).$(SHAREDEXT)  examples/test_knn_linear examples/test_knn_linear_ip test/gpuknn_linear_test

# make dep > x
# then copy x below
dep:
	for i in $(patsubst %.o,%.cpp,$(CPPOBJ)) \
                 $(patsubst %.o,%.cu,$(CUOBJ)); do \
	     echo -n $$( dirname $$i )/ ; \
             cpp -MM -std=gnu++0x $$i; \
	done

./gpu_knn_linear.o : gpu_knn_linear.cpp gpu_knn_linear.h

./GpuResources.o: utils/GpuResources.cpp utils/GpuResources.h utils/DeviceMemory.h \
 utils/DeviceUtils.h utils/FaissAssert.h utils/FaissException.h
./StandardGpuResources.o: utils/StandardGpuResources.cpp utils/StandardGpuResources.h \
 utils/GpuResources.h utils/DeviceMemory.h utils/StackDeviceMemory.h \
 utils/DeviceUtils.h utils/FaissAssert.h utils/FaissException.h utils/FaissAssert.h
./DeviceUtils.o: utils/DeviceUtils.cpp utils/DeviceUtils.h utils/FaissAssert.h utils/FaissException.h
./StackDeviceMemory.o: utils/StackDeviceMemory.cpp \
 utils/StackDeviceMemory.h utils/DeviceMemory.h utils/DeviceUtils.h \
 utils/FaissAssert.h utils/FaissException.h utils/StaticUtils.h
./DeviceMemory.o: utils/DeviceMemory.cpp utils/DeviceMemory.h \
 utils/DeviceUtils.h utils/FaissAssert.h utils/FaissException.h
./MemorySpace.o: utils/MemorySpace.cpp utils/MemorySpace.h \
  utils/FaissAssert.h utils/FaissException.h

./index_distance.o: impl/index_distance.cu impl/selectk.cuh utils/dots.cuh
./linear_index.o: impl/linear_index.cu utils/GpuResources.h impl/index_distance.cuh \
utils/Tensor.cuh utils/Tensor-inl.cuh utils/DeviceVector.cuh \
utils/DeviceTensor.cuh utils/DeviceTensor-inl.cuh 
./index_calculator.o: impl/index_calculator.cu impl/index_calculator.cuh
