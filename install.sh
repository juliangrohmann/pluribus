sudo DEBIAN_FRONTEND=noninteractive apt-get update \
  && sudo DEBIAN_FRONTEND=noninteractive apt-get upgrade -y \
  && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
       vim g++ cmake libsdl2-dev libsdl2-image-dev libsdl2-ttf-dev python3.10-venv libboost-all-dev libtbb-dev libgsl-dev

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
python3 -m venv venv
source venv/bin/activate
pip install wandb
deactivate
cd ..

cd pluribus
mkdir build
mkdir temp

cd build
wget \
  https://pluribus-poker.s3.us-east-1.amazonaws.com/clusters_r{1,2}_c200.npy \
  https://pluribus-poker.s3.us-east-1.amazonaws.com/clusters_r3_c200_p{1,2}.npy

cmake -DCMAKE_BUILD_TYPE=Release -DUNIT_TEST=ON ..
cmake --build .

./Test
./Benchmark