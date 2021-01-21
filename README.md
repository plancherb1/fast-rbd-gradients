# Accelerating Robot Dynamics Gradients on a CPU, GPU, and FPGA

## Timing tests and code for the CPU and GPU.

### The baseline code (and compilation instructions) are:
```
time_CPU_pinnochip.cpp # /utils
clang++-10 -std=c++11 -o pinnochio_timing.exe time_CPU_pinnochio.cpp -O3 -DPINOCCHIO_URDFDOM_TYPEDEF_SHARED_PTR -DPINOCCHIO_WITH_URDFDOM -lboost_system -L/opt/openrobots/lib -lpinocchio -lurdfdom_model -lpthread

time_GPU_WAFR.cu       # /utils, /helpers_WAFR
nvcc -std=c++11 -o WAFR_timing.exe time_GPU_WAFR.cu -gencode arch=compute_75,code=sm_75 -O3
```

### Our accelerated implementations (and compilation instructions) are:
```
time_CPU.cpp           # /utils, /helpers_CPU
g++ -std=c++11 -o CPU_timing.exe time_CPU.cpp -lpthread -O3 -march=native -mavx

time_GPU.cu            # /utils, /helpers_GPU
nvcc -std=c++11 -o GPU_timing.exe time_GPU.cu -gencode arch=compute_75,code=sm_75 -O3
```

## Installing the neccessary software dependencies (for the CPU and GPU timing files)
### Dependencies for the various packages
```
sudo apt-get update
sudo apt-get -y install git build-essential libglib2.0-dev dkms xorg xorg-dev cpufrequtils net-tools linux-headers-$(uname -r) meld apt-transport-https cmake libboost-all-dev
```

### Clang/LLVM
```
sudo bash -c "$(wget -O - https://apt.llvm.org/llvm.sh)"
```

### Eigen
#### Download and install
```
cd ~/Downloads
wget -q http://bitbucket.org/eigen/eigen/get/3.3.7.tar.bz2
tar -xf 3.3.7.tar.bz2
cd eigen*
mkdir build && cd build
cmake ..
sudo make install
```
#### Add symlinks
```
sudo ln -s /usr/local/include/eigen3/Eigen /usr/local/include/Eigen
sudo ln -s /usr/local/include/eigen3/unsupported /usr/local/include/unsupported
```

### Pinnochio
#### Download and install
```
sudo apt install -qqy lsb-release gnupg2 curl
echo "deb [arch=amd64] http://robotpkg.openrobots.org/packages/debian/pub $(lsb_release -cs) robotpkg" | sudo tee /etc/apt/sources.list.d/robotpkg.list
curl http://robotpkg.openrobots.org/packages/debian/robotpkg.key | sudo apt-key add -
sudo apt-get update
sudo apt install -qqy robotpkg-py27-pinocchio
```
#### Add to bashrc
```
echo #Pinnochio >> ~/.bashrc
echo export PATH=/opt/openrobots/bin:$PATH >> ~/.bashrc
echo export PKG_CONFIG_PATH=/opt/openrobots/lib/pkgconfig:$PKG_CONFIG_PATH >> ~/.bashrc
echo export LD_LIBRARY_PATH=/opt/openrobots/lib:$LD_LIBRARY_PATH >> ~/.bashrc
echo export PYTHONPATH=/opt/openrobots/lib/python2.7/site-packages:$PYTHONPATH >> ~/.bashrc
echo export CMAKE_PREFIX_PATH=/opt/openrobots:$CMAKE_PREFIX_PATH >> ~/.bashrc
echo export C_INCLUDE_PATH=/opt/openrobots/include:$C_INCLUDE_PATH >> ~/.bashrc
echo export CPLUS_INCLUDE_PATH=/opt/openrobots/include:$CPLUS_INCLUDE_PATH >> ~/.bashrc
```

### CUDA
#### Download and set things up
```
sudo echo blacklist nouveau > /etc/modprobe.d/blacklist-nouveau.conf
sudo echo options nouveau modeset=0 >> /etc/modprobe.d/blacklist-nouveau.conf
cd ~/Downloads
wget http://developer.download.nvidia.com/compute/cuda/11.0.2/local_installers/cuda_11.0.2_450.51.05_linux.run
sudo update-initramfs -u
```
If you get errors on missing firmware do this: https://askubuntu.com/questions/832524/possible-missing-frmware-lib-firmware-i915
#### Go into headless mode for rest of CUDA install -- GPU MUST BE INSTALLED FOR THIS
```
sudo service lightdm stop # Or Cntrl+Alt+F3 Or boot via recovery mode and enter the root command line
```
```
cd ~/Downloads
sudo sh cuda_11.0.2_450.51.05_linux.run
```
When running it choose to install the drivers and cuda toolkit (no need for the samples or doc)
```
sudo serivce lightdm start # Or Cntrl+Alt+F2 Or Cntrl+d and resume normal boot
```
#### Add to bashrc
Note that cuda below might be cuda-version. In that case update the below or symlink.
```
echo #CUDA >> ~/.bashrc
echo export PATH="usr/local/cuda/bin:$PATH" >> ~/.bashrc
echo export LD_LIBRARY_PATH=":usr/local/cuda/lib64:$LD_LIBRARY_PATH" >> ~/.bashrc
```
