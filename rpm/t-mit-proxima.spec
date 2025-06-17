Name:       t-mit-proxima
Packager:   jiye
Version:1.0.5
Release: %(echo $RELEASE)%{?dist}
Group: MIT
License: Commercial
BuildArch: x86_64
%define _prefix /usr/local
Prefix: %{_prefix}
BuildRoot: %{_tmppath}/%{name}-%{version} AutoReqProv: no
URL: http://gitlab.alibaba-inc.com/vectors-retrieval/proxima
Summary: proxima

%description
proxima for ha3,dii 

%prep
cd $OLDPWD/../
rm -rf static_build
mkdir -p static_build
cd static_build
cmake -DCMAKE_BUILD_TYPE=Release -DENABLE_SSE4.2=ON -DBUILD_STATIC_LIBS=ON ../

cd ../
rm -rf shared_build 
mkdir -p shared_build
cd shared_build
cmake -DCMAKE_BUILD_TYPE=Release -DENABLE_SSE4.2=ON ../

%build
cd $OLDPWD/../static_build
make VERBOSE=1 -j

cd ../shared_build
make VERBOSE=1 -j

%install 
cd $OLDPWD/../static_build
mkdir -p %{buildroot}/%{_prefix}/
mkdir -p %{buildroot}/%{_prefix}/bin
mkdir -p %{buildroot}/%{_prefix}/lib64
mkdir -p %{buildroot}/%{_prefix}/include
mkdir -p %{buildroot}/%{_prefix}/include/proxima/build_adapter/
mkdir -p %{buildroot}/%{_prefix}/include/proxima/common/

cp deps/aitheta/src/src/aitheta/ %{buildroot}/%{_prefix}/include/ -r
cp ../tools/build_adapter/*.h %{buildroot}/%{_prefix}/include/proxima/build_adapter/
cp ../src/common/*.h %{buildroot}/%{_prefix}/include/proxima/common/
find %{buildroot}/%{_prefix}/include/ -name "*.cc" -exec rm -f {} \; -print
cp lib/libknn_*.a lib/libaitheta.a lib/libcluster_onepass.a lib/libbuild_adapter.a %{buildroot}/%{_prefix}/lib64/

cd ../shared_build
cp lib/*.so %{buildroot}/%{_prefix}/lib64/
cp bin/knn_* %{buildroot}/%{_prefix}/bin
cp bin/cluster_* %{buildroot}/%{_prefix}/bin
cp bin/bench %{buildroot}/%{_prefix}/bin
cp bin/recall %{buildroot}/%{_prefix}/bin

%clean
[ "X%{buildroot}" != "X" ] && rm -rf "%{buildroot}"

%files
%{_prefix}/

%changelog
* Fri Jul 06 2018 继业 <jiye@taobao.com>
- Release 1.0.3
