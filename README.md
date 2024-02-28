# ClearGraspPlus: Enhancing depth estimation for transparent objects

## Info
* ClearGraspPlus leverages deep learning with synthetic training data to infer accurate 3D geometry of transparent objects from a single RGB-D image. The estimated geometry can be directly used for downstream robotic manipulation tasks (e.g. suction and parallel-jaw grasping).
* This project was a research work for my AI Master's thesis work titled **Enhancing depth estimation for transparent objects** carried out at [University of Groningen](https://www.rug.nl/masters/artificial-intelligence/?lang=en) under the supervision of [Prof. Dr. Hamidreza](https://hkasaei.github.io/#research) and [Prof. Dr. Matias](https://mvaldenegro.github.io/)
* This repository is a modification of the original work proposed by Shreeyak Sajjan et al. named [clearGrasp](https://github.com/Shreeyak/cleargrasp) and it provides:
  * An API for the depth estimation pipeline
  * PyTorch code for training and testing our models
  * Demo code to see ClearGraspPlus in action with a RealSense D400 series camera
  * Real images dataset capture utility available in another repo named [IRL_transparent_objects_set](https://github.com/AbhishekRS4/IRL_transparent_objects_set) for kincet v1 RGBD camera
* Authors: [Abhishek R. S.](https://abhishekrs4.github.io/), [Prof. Dr. Hamidreza](https://hkasaei.github.io/#research) and [Prof. Dr. Matias](https://mvaldenegro.github.io/)
* The original dataset can be downloaded from here
  * [Download Data - Training Set](https://storage.googleapis.com/cleargrasp/cleargrasp-dataset-train.tar)  
  * [Download Data - Testing and Validation Set](https://storage.googleapis.com/cleargrasp/cleargrasp-dataset-test-val.tar)
  * [Additional Training dataset COCO](http://images.cocodataset.org/zips/train2014.zip)
  * [Additional Training dataset NYU_v2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)
  * [Additional Data for testing](https://github.com/AbhishekRS4/IRL_transparent_objects_set/tree/main/data)
  * [Download 3D Models](https://storage.googleapis.com/cleargrasp/cleargrasp-3d-models.tar)
* In this work, we propose to use a custom network for the individual tasks, an encoder-decoder based network with ResNet50+PSA and DeepLabV3+ respectively, and then use their predictions in the clearGrasp pipeline for enhancing the depth estimation for transparent objects

## Contact
* If you have any questions or find any bugs, please file a github issue or contact me:  
Abhishek R. S.: abhishek[dot]r[dot]satyanarayana[dot]4[at]gmail[dot]com

## Note
* Added some notebooks that were used for creating visualizations. Reference only, might not run. Some modifications might be required to run them.

## Installation
* This code is tested with Ubuntu 20.04, Python 3.8 and [Pytorch](https://pytorch.org/get-started/locally/) v1.12.1, and CUDA v11.3

## System and framework dependencies
* The required system dependencies can be installed using the following commands
```bash
sudo apt-get install libhdf5-10 libhdf5-serial-dev libhdf5-dev libhdf5-cpp-11
sudo apt install libopenexr-dev zlib1g-dev openexr # required for openexr
sudo apt install xorg-dev  # display widows
sudo apt install libglfw3-dev
```
* The required PyTorch version can be installed with the following command
```
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```
* For `openexr` python package, install it using the following command `pip install --no-binary openexr openexr`
* The rest of the Python package dependencies can be found in [requirements.txt](requirements.txt)

## LibRealSense (Optional)
* If you want to run demos with an Intel RealSense camera, you may need to install [LibRealSense](https://github.com/IntelRealSense/librealsense). It is required to stream and capture images from Intel Realsense D415/D435 stereo cameras. Please check the [installation guide](https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md) to install from binaries, or compile from source.
```bash
# Register the server's public key:
$ sudo apt-key adv --keyserver keys.gnupg.net --recv-key C8B3A55A6F3EFCDE || sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key C8B3A55A6F3EFCDE

# Ubuntu 16 LTS - Add the server to the list of repositories
$ sudo add-apt-repository "deb http://realsense-hw-public.s3.amazonaws.com/Debian/apt-repo xenial main" -u

# Install the libraries
$ sudo apt-get install librealsense2-dkms
$ sudo apt-get install librealsense2-utils

# Install the developer and debug packages
$ sudo apt-get install librealsense2-dev
$ sudo apt-get install librealsense2-dbg
```

## Setup

1. Clone the repository. A small sample dataset of 3 real and 3 synthetic images is included.

2. Install all the dependencies as mentioned in the **System and framework dependencies** section.

3. Download the data:  
   a) [Train dataset](https://storage.googleapis.com/cleargrasp/cleargrasp-dataset-train.tar) (Optional, 72GB) - Contains the synthetic images used for training the models. No real images were used for training.  
   b) [Val + Test datasets](https://storage.googleapis.com/cleargrasp/cleargrasp-dataset-test-val.tar) (Optional, 1.7GB) - Contains the real and synthetic images used for validation and testing.  

4. Compile depth2depth (global optimization):
  * `depth2depth` is a C++ global optimization module used for depth completion, adapted from the [DeepCompletion](http://deepcompletion.cs.princeton.edu/) project. It resides in the `api/depth2depth/` directory.

  * To compile the depth2depth binary, you will first need to identify the path to libhdf5. Note the location of `hdf5/serial`. Run the following
  command in terminal:
    ```bash
    find /usr -iname "*hdf5.h*"
    ```
    It will look similar to: `/usr/include/hdf5/serial/hdf5.h`.

  * Edit BOTH lines 28-29 of the makefile at `api/depth2depth/gaps/apps/depth2depth/Makefile` to add the path you just found as shown below:
    ```bash
    USER_LIBS=-L/usr/include/hdf5/serial/ -lhdf5_serial
    USER_CFLAGS=-DRN_USE_CSPARSE "/usr/include/hdf5/serial/"
    ```

  * Compile the binary:
    ```bash
    cd api/depth2depth/gaps
    export CPATH="/usr/include/hdf5/serial/"  # Ensure this path is same as read from output of `find /usr -iname "*hdf5.h*"`

    make
    ```
    This should create an executable, `api/depth2depth/gaps/bin/x86_64/depth2depth`. The config files will need the path to this executable to run the  depth estimation pipeline.

  * Check the executable, by passing in the provided sample files:
    ```bash
    cd api/depth2depth/gaps
    bash depth2depth.sh
    ```
    This will generate `gaps/sample_files/output-depth.png`, which should match the `expected-output-depth.png` sample file. It will also generate RGB visualizations of all the intermediate files.


## To run the code:

### 1. ClearGrasp Quick Demo - Evaluation of Depth Completion of Transparent Objects
* We provide a script to run our full pipeline on a dataset and calculate accuracy metrics (RMSE, MAE, etc). Resides in the directory `eval_depth_completion/`.  

* Create a local copy of the config file:
  ```bash
  cd eval_depth_completion/
  cp config/config.yaml.sample config/config.yaml
  ```

* Edit the `config/config.yaml` file to set `pathWeightsFile` parameters to the paths of the respective model checkpoints. To run evaluation on the different datasets, set the path(s) to their director(ies) within the `files` parameter.

* Run ClearGrasp on the sample dataset:
  ```bash
  python eval_depth_completion.py -c config/config.yaml
  ```

* The script will run ClearGrasp on the given dataset, storing all it's output and calculating accuracy metrics of the depth completion of transparent objects. The metrics (RMSE, etc.) are stored in a csv file in the results dir. Resized inputs, output depths, output pointclouds and other intermediate files are also saved.



### 2. Live Demo

* We provide a demonstration of how to use our API on images streaming from realsense D400 series camera. Each new frame coming from the camera stream is passed through the depth completion module to obtain completed depth of transparent objects and the results are displayed in a window.  
Resides in the folder `live-demo/`. This demo requires the Librealsense SDK to be installed.

  * Create a copy of the sample config file:
    ```bash
    cd live_demo
    cp config/config.yaml.sample config/config.yaml
    ```

  * Edit `config.yaml` with paths to checkpoints of networks and depth2depth executable. Edit parameters as per your camera.

  * Compile `realsense.cpp`:
    ```bash
    cd live-demo/realsense/
    mkdir build
    cd build
    cmake ..
    make
    ```
    This will create a binary `build/realsense` which is used to stream images from the realsense camera over TCP/IP. In case of issues, check FAQ.

  * Connect a realsense d400 series camera to USB and start the camera stream:
    ```bash
    cd live_demo/realsense
    ./build/realsense
    ```
    This application will capture RGB and Depth images from the realsense and stream them on an TCP/IP port. It will also open a window with the RGB and Depth images displayed.

  * Run demo:
    ```bash
    python live_demo.py -c config/config.yaml
    ```
    This will open a new window displaying input image, input depth, intermediate outputs (surface normals, occlusion boundaries, mask), modified input depth and output depth. Expect around 1 FPS with an i7 7700K CPU and 1080ti GPU. The global optimization module is CPU bound and takes almost 1 sec per image at 256x144p resolution with CPU at 4.2GHz.

### 3. Training Code

* The folder `pytorch_networks/` contains the code used to train the
surface normals, occlusion boundary and semantic segmentation models.

* Go the to respective folder (eg: `pytorch_networks/surface_normals`) and create a local copy of the config file:
  ```bash
  cp config/config.yaml.sample config/config.yaml
  ```

* Edit the `config.yaml` file to fill in the paths to the dataset, select hyperparameter values, etc. All the parameters are explained in comments within the config file.
* Start training:
  ```bash
  python train.py -c config/config.yaml
  ```
* Eval script can be run by:
  ```bash
  python eval.py -c config/config.yaml
  ```

### 4. Dataset Capture
* Contains GUI application that was used to collect dataset of real transparent objects. First the transparent objects were placed in the scene along with various random opaque objects like cardboard boxes, decorative mantelpieces and fruits. After capturing and freezing that frame, each object was replaced with an identical spray-painted instance. Subsequent frames would be overlaid on the frozen frame so that the overlap between the
spray painted objects and the transparent objects they were replacing could be observed. With high resolution images, sub-millimeter accuracy can be achieved in the positioning of the objects.
* Run the `dataset_capture_gui/capture_image.py` script to launch a window that streams images directly from a Realsense D400 series camera. Press 'c' to capture the transparent frame, 'v' to capture the opaque frame and spacebar to confirm and save the RGB and Depth images for both frames.
* The additional dataset captured and the scripts used are available in another repository named [IRL_transparent_objects_set](https://github.com/AbhishekRS4/IRL_transparent_objects_set) for kincet v1 RGBD camera


## FAQ

### Details on depth2depth
* The `depth2depth` executable expects the following parameters:
  * **input_depth.png**: The path for the raw depth map from sensor, which is the depth to refine. It should be saved as 4000 x depth in meter in a 16bit PNG.
  * **output_depth.png**: The path for the result, which is the completed depth. It is also saved as 4000 x depth in meter in a 16bit PNG.
  * **Occlusion Weights**: The depth discontinuities channel is extracted from the occlusion outlines models' outputs scaled and saved as png file.
  * **Surface Normals**: The output of surface normals model is saved as an .h5 file
  * **xres, yres**: The resolution of image in x and y axes.
  * **fx, fy**: The focal length used to take image in pixels
  * **cx, cy**: The centre of the image. Ideally it is equal to (height/2, width/2)
  * **inertia weight**: The strength of the penalty on the difference between the input and the output depth map on observed pixels. Set this value higher if you want to maintain the observed depth from input_depth.png.
  * **smoothness_weight**: The strength of the penalty on the difference between the depths of neighboring pixels. Higher smoothness weight will produce soap-film-like result.
  * **tangent_weight**: The universal strength of the surface normal constraint. Higher tangent weight will force the output to have the same surface normal with the given one.

### Calculation of focal len in pixels (fx, fy)
* The focal len in pixels is calculated from the Field of View and Sensor Size of camera, as derived from [here](https://photo.stackexchange.com/questions/97213/finding-focal-length-from-image-size-and-fov):
  ```bash
  F = A / tan(a)
    Where,
      F = Focal len in pixels
      A = image_size/2
      a = FOV/2

  => (focal len in pixels) = ((image width or height)/2 ) / tan( FOV/2 )
  ```
* Here are the calculation for our synthetic images, with angles in degrees for image output at 288x512p:
  ```bash
  Fx = (512 / 2) / tan( 69.40 / 2 ) = 369.71 = 370 pixels
  Fy = (288 / 2) / tan( 42.56 / 2 ) = 369.72 = 370 pixels
  ```

### Notes on data:
* The 4x4 transformation matrix for each object in the scene can give incorrect rotations since it is not normalized. Use the provided quaternion to get the rotation of each object.
* Some objects are present in the scene, but not visible to the camera. Your code will have to account for such objects when parsing through the data, using the provided masks.

### ERROR: No module named open3d
* In case of Open3D not being recognized, try installing with:
  ```bash
  pip uninstall open3d-python
  pip uninstall open3d
  pip install open3d --no-cache-dir
  ```

### FIX for librealsense version V2.15 and earlier
* Change the below line:

  ```c
  // Find and colorize the depth data
  rs2::frame depth_colorized = color_map.colorize(aligned_depth);
  ```

  to

  ```c
  // Find and colorize the depth data
  rs2::frame depth_colorized = color_map(aligned_depth);
  ```

### ERROR: depth2depth.cpp:11:18: fatal error: hdf5.h: No such file or directory
* Make sure HDF5 is installed.
* Ensure you edited both lines in the makefile to add path to hdf5, as per directions in Installation section.  
* Make sure you exported CPATH before compiling `depth2depth`, as mentioned above (`export CPATH="/usr/include/hdf5/serial/"`).

### ERROR: /usr/bin/ld: cannot find -lrealsense2
* You may face this error when compiling realsense.cpp. This may occur when using later versions of librealsense (>=2.24, circa Jun 2019).  
* This error can be resolved by compiling Librealsense from source. Please follow the [official instructions](https://github.com/IntelRealSense/librealsense/blob/master/doc/installation.md).

### HOW to change the image resolution streaming from realsense camera?

* You can change the image resolution by changing the corresponding lines in the `live_demo/realsense/realsense.cpp` file and re-compiling realsense:
  ```shell
  int stream_width = 640;
  int stream_height = 360;
  int depth_disparity_shift = 25;
  int stream_fps = 30;
  ```
* Also change the following lines in the `live_demo/realsense/camera.py` file to match the cpp file:
  ```shell
  self.im_height = 360
  self.im_width = 640
  self.tcp_host_ip = '127.0.0.1'
  self.tcp_port = 50010
  ```
