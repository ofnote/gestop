### gestures-mediapipe


Built on top of mediapipe, this repo aims to be a tool to navigate a computer through gestures. The hand keypoints are detected using google's mediapipe. These keypoints are then fed into a Python script for further use through ZeroMQ and protobuf. 

The keypoint of the tip of the index finger is extracted in the Python script, and, through pyautogui, used to control the mouse. [WIP]

### Note

As well as mediapipe's own requirements, there a few other things required for this project.

1. ZeroMQ (along with cppzmq and pyzmq)

The zeromq library (libzmq.so) must be symlinked into this directory. The header only C++ binding cppzmq must also be installed and its header (zmq.hpp) symlinked into the working directory. The python module pyzmq must also be installed for python to read the keypoints being sent.

2. protobuf (Python)

The protobuf module in Python must be installed, through the use of `pip` or your distribution's package manager.

3. pyautogui

pyautogui is a GUI automation python module used, in this case, to simulate the movement of the mouse. As with other python packages, to be installed through `pip` or package manager e.g. `apt`. 

### Usage

1. Clone mediapipe and set it up. Make sure the provided hand tracking example is working.
2. Clone this repo in the top level directory of mediapipe. Install all dependencies
3. Run the instructions below to build and then execute the code. Note: Run build instructions in the mediapipe/ directory, not inside this directory.

# Mediapipe Executable

`    bazel build -c opt --verbose_failures --copt -DMESA_EGL_NO_X11_HEADERS --copt -DEGL_NO_X11 gestures-mediapipe:hand_tracking_gpu       `

`    GLOG_logtostderr=1 bazel-bin/gestures-mediapipe/hand_tracking_gpu --calculator_graph_config_file=gestures-mediapipe/hand_tracking_desktop_live.pbtxt `

# Python Script

`   python py_landmarks.py  `