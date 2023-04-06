# Kinect v1 driver script

## Info
* The script [capture_images.py](capture_images.py) can be used to capture RGB and depth images with kinect v1.

## Setup instructions
* The following are instructions to install the dependency - [libfreenect](https://github.com/OpenKinect/libfreenect)
1. Install cython
```
sudo apt-get install cython3
```
2. Build and install libfreenect
```
git clone https://github.com/OpenKinect/libfreenect
cd libfreenect
mkdir build
cd build
cmake .. -L -DBUILD_PYTHON3=ON -DPython3_EXACTVERSION=3.8.10 -DCYTHON_EXECUTABLE=/usr/bin/cython3
make
sudo make install
```
3. Add environment variables to `bashrc`
```
sudo vim ~/.bashrc
export LD_PRELOAD="/usr/local/lib/libfreenect.so"
source ~/.bashrc
```
4. Add the following udev rules to `/etc/udev/rules.d/51-kinect.rules`
```
sudo adduser $USER video
sudo vim /etc/udev/rules.d/51-kinect.rules

# ATTR{product}=="Xbox NUI Motor"
SUBSYSTEM=="usb", ATTR{idVendor}=="045e", ATTR{idProduct}=="02b0", MODE="0666"
# ATTR{product}=="Xbox NUI Audio"
SUBSYSTEM=="usb", ATTR{idVendor}=="045e", ATTR{idProduct}=="02ad", MODE="0666"
# ATTR{product}=="Xbox NUI Camera"
SUBSYSTEM=="usb", ATTR{idVendor}=="045e", ATTR{idProduct}=="02ae", MODE="0666"
# ATTR{product}=="Xbox NUI Motor"
SUBSYSTEM=="usb", ATTR{idVendor}=="045e", ATTR{idProduct}=="02c2", MODE="0666"
# ATTR{product}=="Xbox NUI Motor"
SUBSYSTEM=="usb", ATTR{idVendor}=="045e", ATTR{idProduct}=="02be", MODE="0666"
# ATTR{product}=="Xbox NUI Motor"
SUBSYSTEM=="usb", ATTR{idVendor}=="045e", ATTR{idProduct}=="02bf", MODE="0666"
```
5. Add the following udev rules to `/etc/udev/rules.d/66-kinect.rules`
```
sudo vim /etc/udev/rules.d/66-kinect.rules

SYSFS{idVendor}=="045e", SYSFS{idProduct}=="02ae", MODE="0660",GROUP="video"
SYSFS{idVendor}=="045e", SYSFS{idProduct}=="02ad", MODE="0660",GROUP="video"
SYSFS{idVendor}=="045e", SYSFS{idProduct}=="02b0", MODE="0660",GROUP="video"
```
6. Install python wrappers for libfreenect
```
cd wrappers/python
python3 setup.py install
```
7. To check if the Kinect v1 is detected, run the following command and the following result must be the output of the command
```
lsusb | grep Xbox
Bus 001 Device 021: ID 045e:02ae Microsoft Corp. Xbox NUI Camera
Bus 001 Device 019: ID 045e:02b0 Microsoft Corp. Xbox NUI Motor
Bus 001 Device 020: ID 045e:02ad Microsoft Corp. Xbox NUI Audio
```


## Script usage
* This script can be used only on Linux environment with the above mentioned dependencies.
* To list all the commandline options, run the following command
```
source ~/.bashrc
python3 capture_images.py --help
```
* To capture data with kinect v1, run
```
source ~/.bashrc
python3 capture_images.py
```
* Use `Ctrl + C` to stop the script.
