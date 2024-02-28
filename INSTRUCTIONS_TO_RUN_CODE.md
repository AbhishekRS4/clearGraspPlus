# Instructions to setup dependencies and run code


1) To Setup dependencies

	a) sudo apt-get install libhdf5-10 libhdf5-serial-dev libhdf5-dev libhdf5-cpp-11
	b) sudo apt install libopenexr-dev zlib1g-dev openexr # required for openexr
	c) For `openexr` python package, install it using the following command `pip install --no-binary openexr openexr`
	d) The rest of the Python package dependencies can be found in [requirements.txt](requirements.txt)

2) To build depth2depth module
	a) To compile the depth2depth binary, you will first need to identify the path to libhdf5. Note the location of `hdf5/serial`. Run the following
  command in terminal:
    ```bash
    find /usr -iname "*hdf5.h*"
    ```
    It will look similar to: `/usr/include/hdf5/serial/hdf5.h`.

	b) Edit BOTH lines 28-29 of the makefile at `api/depth2depth/gaps/apps/depth2depth/Makefile` to add the path you just found as shown below:
    ```bash
    USER_LIBS=-L/usr/include/hdf5/serial/ -lhdf5_serial
    USER_CFLAGS=-DRN_USE_CSPARSE "/usr/include/hdf5/serial/"
    ```

	c) Compile the binary:
    ```bash
    cd api/depth2depth/gaps
    export CPATH="/usr/include/hdf5/serial/"  # Ensure this path is same as read from output of `find /usr -iname "*hdf5.h*"`

    make
    ```
    This should create an executable, `api/depth2depth/gaps/bin/x86_64/depth2depth`. The config files will need the path to this executable to run the  depth estimation pipeline.

	d) Check the executable, by passing in the provided sample files:
    ```bash
    cd api/depth2depth/gaps
    bash depth2depth.sh
    ```
    This will generate `gaps/sample_files/output-depth.png`, which should match the `expected-output-depth.png` sample file. It will also generate RGB visualizations of all the intermediate files.

3) To Run code
	a) Object segmentation
		i) All the scripts can be found inside `pytorch_networks/object_segmentation` directory.
		ii) The config for testing can be found in `config_test` directory in the above directory, with the name `config_resnet50_psa.yaml`. The appropriate model weights file and dataset paths needs to be set in the yaml file.
		iii) For evaluation, `evaluate.py` needs to be used. This script requires that the dataset has groundtruth files.
		iv) For saving prediction visualizations, `save_prediction_vis.py` needs to be used. This script requires only the input RGB images.


	b) Occlusion boundary
		i) All the scripts can be found inside `pytorch_networks/occlusion_boundary` directory.
 		ii) The config for testing can be found in `config_test` directory in the above directory, with the name `config_resnet50_psa.yaml`. The appropriate model weights file and dataset paths needs to be set in the yaml file.
		iii) For evaluation, `evaluate.py` needs to be used. This script requires that the dataset has groundtruth files.
		iv) For saving prediction visualizations, `save_prediction_vis.py` needs to be used. This script requires only the input RGB images.

	c) Surface normal
		i) All the scripts can be found inside `pytorch_networks/surface_normals` directory.
		ii) The config for testing can be found in `config_test` directory in the above directory, with the name `config_resnet50_psa.yaml`. The appropriate model weights file and dataset paths needs to be set in the yaml file.
		iii) For evaluation, `evaluate.py` needs to be used. This script requires that the dataset has groundtruth files.
		iv) For saving prediction visualizations, `save_prediction_vis.py` needs to be used. This script requires only the input RGB images.

	d) Depth estimation
		i) All scripts can be found inside `eval_depth_completion` directory.
		ii) The config for testing can be found in `config` directory in the above directory, with the name `config_resnet50_psa_real.yaml`. The appropriate model weights file and dataset paths needs to be set in the yaml file.
		iii) For evaluation, `eval_depth_completion.py` needs to be used. This script requires that the dataset has groundtruth files.
		iv) For saving prediction visaulizations, `run_depth_completion.py` needs to be used. This script requires only the input RGB images.
