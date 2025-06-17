#!/bin/bash 
set -x
set -e

#yum install -y --nogpgcheck flex flex-devel bison zlib zlib-devel
TYPE=${type}
mkdir -p deploy
cd deploy
cmake -DCMAKE_BUILD_TYPE=Release -D ENABLE_SSE4.2=ON ../
make -j 8

if [ ! -f ./bin/centroid_trainer ] ; then
    echo "no centroid_trainer file."
    exit 255
fi 

if [ ! -f ./bin/local_builder ] ; then
    echo "no local_builder file."
    exit 255
fi 

if [ ! -f ./lib/libframework.so ] ; then
    echo "no libframework.so file."
    exit 255
fi 

cd ..
OUTPUTDIR=output-mercury

rm -rf ${OUTPUTDIR}
mkdir -p ${OUTPUTDIR}
mkdir -p ${OUTPUTDIR}/lib

cp -df ./deploy/lib/*.so ${OUTPUTDIR}/lib/
cp -df ./deploy/bin/centroid_trainer ${OUTPUTDIR}/
cp -df ./deploy/bin/local_builder ${OUTPUTDIR}/
