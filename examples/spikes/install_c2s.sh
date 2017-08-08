#!/bin/sh

apt-get install autoconf automake libtool
wget https://github.com/lucastheis/cmt/archive/develop.zip
unzip develop.zip && rm develop.zip
cd cmt-develop/code/liblbfgs
sh autogen.sh
./configure --enable-sse2
make CFLAGS="-fPIC"
cd ../..
pip install scipy numpy
python setup.py build
python setup.py install
rm cmt-develop
pip install git+https://github.com/lucastheis/c2s.git
