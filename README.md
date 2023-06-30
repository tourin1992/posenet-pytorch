## PoseNet Pytorch with MobileNet-v1

### Install

Assumig that the conda package manager is installed, a fresh conda Python 3.6/3.7 environment with the following installs should suffice: 
```
conda install -c pytorch pytorch cudatoolkit
pip install requests opencv-python
```

### Usage

Execute in the terminal while being in the project folder ~/../posenet-pytorch/
```
python webcam_demo.py --model=101
```
After the first run the algorithm will download the corresponding modeland this process might take a few minutes, for the next runs it will already use the downloaded model.

For the higher fps one can use `--model=50` however the overall accuracy will be lower.

The thickness value of the skeleton lines can be adjusted with the `--line_thickness` argument, which default value is 3.

The webcam demo uses OpenCV to capture images from a connected webcam. The result is overlayed with the keypoints and skeletons and rendered to the screen. The default args for the webcam_demo assume `device_id=0` for the camera and that 1280x720 resolution is possible, if some other cameras are connected one might consider chanhing the `device_id` to 1,2 etc.


