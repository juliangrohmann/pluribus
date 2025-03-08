apt update
apt upgrade
apt install g++ cmake libsdl2-dev libsdl2-image-dev
apt-get install libboost-all-dev
apt-get install libtbb-dev
apt-get install libsdl2-ttf-dev

git clone https://github.com/catchorg/Catch2.git
cd Catch2
cmake -B build -S . -DBUILD_TESTING=OFF
cmake --build build/ --target install
cd ..

git clone https://github.com/rogersce/cnpy.git
cd cnpy
mkdir build
cd build
cmake ..
make
make install
cd ../..

cd pluribus
mkdir build
cd build
cmake -DVERBOSE=OFF -DUNIT_TEST=ON ..
cmake --build .

# TODO: download clusters

./Test
./Benchmark