# LVE - 向量检索项目 

## How to Build?

#### 1. REQUIRED

```
 gcc >= 4.8.5
 cmake >= 2.8.12
```

#### 2. COMMAND

* Build a debug version and run all unit tests:

```shell
 mkdir build
 cd build
 cmake -DCMAKE_BUILD_TYPE=Debug ..
 make unittest
```

* Build a release version and run all unit tests:

```shell
 mkdir build
 cd build
 cmake -DCMAKE_BUILD_TYPE=Release ..
 make unittest
```

* Only build the library:

```shell
 mkdir build
 cd build
 cmake -DCMAKE_BUILD_TYPE=Debug ..
 make all
```

* Gather the unit tests code coverage:

```shell
 mkdir build
 cd build
 cmake -DCMAKE_BUILD_TYPE=Debug -D ENABLE_COVERAGE=ON ..
 make unittest
 ../scripts/gcov.sh -tgcov
```

#### 3. OPTIONS

* Cache entries of cmake command:

```
 BUILD_SHARED_LIBS:  Build as a shared library
 ENABLE_M32:         Enable 32-bit platform cross build
 ENABLE_M64:         Enable 64-bit platform cross build
 ENABLE_COVERAGE:    Enable code coverage
 ENABLE_OPENMP:      Enable OpenMP support
 ENABLE_SSE:         Enable Intel SSE instructions
 ENABLE_SSE2:        Enable Intel SSE2 instructions
 ENABLE_SSE3:        Enable Intel SSE3 instructions
 ENABLE_SSSE3:       Enable Intel SSSE3 instructions
 ENABLE_SSE4.1:      Enable Intel SSE4.1 instructions
 ENABLE_SSE4.2:      Enable Intel SSE4.2 instructions
 ENABLE_AVX :        Enable Intel AVX instructions
 ENABLE_AVX2:        Enable Intel AVX2 instructions
 ENABLE_AVX512F:     Enable Intel AVX512F instructions
 ENABLE_FMA :        Enable Intel FMA instructions
```
