# gestop

Built on top of [mediapipe](https://github.com/google/mediapipe), this project aims to be a tool to interact with a computer through hand gestures. Out of the box, using this tool, it is possible to:

1. Use your hand to act as a replacement for the mouse.
2. Perform hand gestures to control system parameters like screen brightness, volume etc.

However, it is possible to extend and customize the functionality of the application in numerous ways:

1. Remap existing hand gestures to different functions in order to better suit your needs.
2. Create custom functionality through the use of either python functions or shell scripts.
3. Collect data and create your own custom gestures to use with existing gestures. 

### [Demo video link](https://www.youtube.com/watch?v=K2UkIjK7BTI&t=19s)

#### [Models link](https://drive.google.com/drive/folders/16lbPkdYWcmLfx0oFo01A5FwTiCDK5DDK?usp=sharing)

#### [Dataset link](https://drive.google.com/drive/folders/1zMFQVKvpAhU-EKGxQNyFXKTu1TgBH23L?usp=sharing)

### Requirements

As well as mediapipe's own requirements, there are a few other libraries required for this project.

* **ZeroMQ** - The zeromq library (*libzmq.so*) must be installed and symlinked into this directory. The header only C++ binding **cppzmq** must also be installed and its header (*zmq.hpp*) symlinked into the directory. 
* **pyzmq**
* **protobuf** 
* **pynput**
* **pytorch**
* **pytorch-lightning**
* **xdotool**

### Usage

1. Clone mediapipe and set it up. Make sure the provided hand tracking example is working.
2. Clone this repo in the top level directory of mediapipe. Install all dependencies.
3. Download the `models/` folder from the link above and place it in the `gestop/` directory.
4. Run the instructions below to build and then execute the code. 

*Note:* Run build instructions in the `mediapipe/` directory, not inside this directory.

*Note:* Python dependencies can be installed simply by creating a virtual environment and running `pip install -r requirements.txt`

#### Mediapipe Executable

##### GPU (Linux only)
``` sh
bazel build -c opt --verbose_failures --copt -DMESA_EGL_NO_X11_HEADERS --copt -DEGL_NO_X11 gestop:hand_tracking_gpu

GLOG_logtostderr=1 bazel-bin/gestop/hand_tracking_gpu --calculator_graph_config_file=gestop/hand_tracking_desktop_live.pbtxt
```

##### CPU
``` sh
bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 gestop:hand_tracking_cpu

GLOG_logtostderr=1 bazel-bin/gestop/hand_tracking_cpu --calculator_graph_config_file=gestop/hand_tracking_desktop_live.pbtxt
```

#### Python Script

``` python
python gestop/gesture_receiver.py
```

### Overview

The hand keypoints are detected using google's mediapipe. These keypoints are then fed into a Python script through zmq.  The tool utilizes the concept of **modes** i.e. the tool is always in one of two modes, either **mouse** or **gestures**. 

The **mouse** mode comprises of all functionality relevant to the mouse, which includes mouse tracking and the various possible mouse button actions. The mouse is tracked simply by moving the hand in mouse mode, where the tip of the index finger reflects the position of the cursor. The gestures related to the mouse actions are detailed below. A dataset was created and a neural network was trained on these gestures and with the use of the python library `pynput`, mouse actions are simulated.

The **gestures** mode is for more advanced dynamic gestures involving a moving hand. It consists of various other actions to interface with the system, such as modifying screen brightness, switching workspaces, taking screenshots etc. The data for these dynamic gestures comes from [SHREC2017 dataset](http://www-rech.telecom-lille.fr/shrec2017-hand/). Dynamic gestures are detected by holding down the `Ctrl` key, performing the gesture, and then releasing the key.

The project consists of a few distinct pieces which are:

* The mediapipe executable - A modified version of the hand tracking example given in mediapipe, this executable tracks the keypoints, stores them in a protobuf, and transmits them using ZMQ.
* Gesture Receive - See `gesture_receiver.py`, responsible for handling the ZMQ stream and utilizing all the following modules.
* Mouse tracking - See `mouse_tracker.py`, responsible for moving the cursor using the position of the index finger.
* Config detection - See `gesture_recognizer.py`, takes in the keypoints from the mediapipe executable, and converts them into a high level description of the state of the hand, i.e. a gesture name.
* Config action - See `gesture_executor.py`, uses the gesture name from the previous module, and executes an action.

### Notes

* Dynamic gestures are only supported with right hand, as all data from SHREC is right hand only.
* A left click can be performed by performing the mouse down gesture and immediately returning to the open hand gesture to register a single left mouse button click.
* For dynamic gestures to work properly, you may need to change the keycodes being used in `gesture_executor.py`. Use the given `find_keycode.py` to find the keycodes of the keys used to change screen brightness and volumee. Finally, system shortcuts may need to be remapped so that the shortcuts work even with the Ctrl key held down. For example, in addition to the usual default behaviour of `<Prnt_Screen>` taking a screenshot, you may need to add `<Ctrl+Prnt_Screen>` as a shortcut as well. 

### Customization

**gestop** is highly customizable and can be easily extended in various ways. The existing gesture-action pairs can be remapped easily, new actions can be defined (either a python function or a shell script, opening up a world of possiiblity to interact with your computer), and finally, if you so desire, you can capture data to create your own gestures and retrain the network to utilize your own custom gestures. The ways to accomplish the above are briefly described in this section. 

The default gesture-action mappings are stored in `data/action_config.json`. The format of the config file is:

`{'gesture_name':['type','func_name']}`

Where, gesture_name is the name of the gesture that is detected, type is either `sh`(shell) or `py`(python). If the type is `py`, then `func_name` is the name of a python function and if the type is `sh`, then `func_name` is either a shell command or a shell script (`./path/to/shell_script.sh`). Refer `data/action_config.json` and `gesture_executor.py` for more details.

It is encouraged to make all custom configuration in a new file rather than replace the old. So, before your modifications, copy `data/action_config.json` and create a new file. After your modifications are done in the new file, you can run the application with your custom config using `python gesture_receiver.py --config-path my_custom_config.json`

#### Remap gestures

To remap functionality, all you need to do is swap the values (i.e. ``['type','func_name']`) for the gestures you wish to remap. As an example if you wish to take a screenshot with `Swipe +`, instead of `Grab`, the configuration would change from:

``` json
    "Grab" : ["py", "take_screenshot"],
    "Swipe +" : ["py", "no_func"],
```

To,

``` json
    "Grab" : ["py", "no_func"],
    "Swipe +" : ["py", "take_screenshot"],
```

#### Adding new actions

Adding new actions is a similar process to remapping gestures, except for the additional step of defining your python function/shell command. As a simple example, if you wish to type your username on performing `Pinch`, the first step would be to write the python function in `user_config.py`. The function would be something similar to the following:

``` python
def print_username(self, S):
    ''' Prints username '''
    self.keyboard.type("sriramsk1999")
    return S
```

Where `S` represents the *State* and is passed to all the functions in `user_config.py`. Refer `user_config.py` and `config.py` to see more examples of how to add new actions. 

Finally, replace the existing `Pinch` mapping with your own in your configuration file.

``` json

    "Pinch" : ["py", "print_username"],
```

#### Adding new gestures

To extend this application and create new gestures, there are a few prerequisites. Firstly, download the data from from the dataset link given above, and place in the `data/` directory. This is to ensure that your model has all existing data along with the new data to train on.

You can either record a new static gesture or a dynamic gesture, with the `static_data_collection.py` and `dynamic_data_collection.py` scripts respectively.

To collect data for a new static gesture, run the program, enter the name of the gesture and the hand with which you will be performing the gesture. Run the mediapipe executable and hold the gesture while data is collected. A 1000 samples are collected which should take a minute or two. Hold your hand in the same pose in good lighting to ensure the model gets clean data.

To collect data for a new dynamic gesture, the process is mostly similar. Run the `dynamic_data_collection.py` program, enter the name of the gesture and run the mediapipe executable. Data is collected only when the Ctrl key is held down, so to collect a single sample, hold the ctrl key, perform the gesture and then release. Repeat this process a few dozen times to collect enough data.

Finally, retrain the network. Use the `static_train_model.py` or the `dynamic_train_model.py` script depending on the new gesture. And after that, your new gesture is ready to be used in the configuration file.

### Gestures

#### Static Gestures

| Gesture name   | Gesture Action   | Image                               |
| -------------- | ---------------- | --------------------------------    |
| seven          | Left Mouse Down  | ![seven](images/seven2.png)         |
| eight          | Double Click     | ![eight](images/eight2.png)         |
| four           | Right Mouse Down | ![four](images/four2.png)           |
| spiderman      | Scroll           | ![spiderman](images/spiderman2.png) |
| hitchhike      | Mode Switch      | ![hitchhike](images/hitchhike2.png) |

#### Dynamic Gestures

| Gesture name             | Gesture Action                     | Gif                                              |
| --------------           | ----------------                   | ---------                                        |
| Swipe Right              | Move to the workspace on the right | ![swiperight](images/swiperight.gif)             |
| Swipe Left               | Move to the workspace on the left  | ![swipeleft](images/swipeleft.gif)               |
| Swipe Up                 | Increase screen brightness         | ![swipeup](images/swipeup.gif)                   |
| Swipe Down               | Decrease screen brightness         | ![swipedown](images/swipedown.gif)               |
| Rotate Clockwise         | Increase volume                    | ![clockwise](images/clockwise.gif)               |
| Rotate Counter Clockwise | Decrease volume                    | ![counterclockwise](images/counterclockwise.gif) |
| Grab                     | Screenshot                         | ![grab](images/grab.gif)       |
| Tap                      | Mode Switch                        | ![tap](images/tap.gif)                           |

<details>
<summary><b>Repo Overview</b></summary>
<br>
<ul>
<li> models -> Stores the trained model(s) which can be called by other files for inference </li>
<li> proto -> Holds the definitions of the protobufs used in the project for data transfer</li>
<li> BUILD -> Various build instructions for Bazel</li>
<li> <code>static_data_collection.py</code> -> Script to create a custom static gesture dataset.  </li>
<li> <code>dynamic_data_collection.py</code> -> Script to create a custom dynamic gesture dataset.  </li>
<li> <code>data/static_gestures_mapping.json</code> -> Stores the encoding of the static gestures as integers</li>
<li> <code>data/dynamic_gestures_mapping.json</code> -> Stores the encoding of the dynamic gestures as integers</li>
<li> <code>data/static_gestures_data.csv</code> -> Dataset created with data_collection.py </li>
<li> <code>data/action_config.json</code> -> Configuration of what gesture maps to what action. </li>
<li> <code>hand_tracking_desktop_live.pbtxt</code> -> Definition of the mediapipe calculators being used. Check out mediaipe for more details.</li>
<li> <code>hand_tracking_landmarks.cc</code> -> Source code for the mediapipe executable. GPU version is Linux only.</li>
<li> <code>model.py</code> -> Definition of the models used.</li>
<li> <code>static_train_model.py</code> -> Trains the "GestureNet" model for static gestures and saves to disk</li>
<li> <code>dynamic_train_model.py</code> -> Trains the "ShrecNet" model for dynamic gestures and saves to disk</li>
<li> <code>find_keycode.py</code> -> A sample program from pynput used to find the keycode of the key that was pressed. Useful in case the brightness and audio keys vary.</li>
<li> <code>gesture_receiver.py</code> -> Handles the stream of data coming from the mediapipe executable by passing it to the various other modules.</li>
<li> <code>mouse_tracker.py</code> -> Functions which implement mouse tracking.</li>
<li> <code>gesture_recognizer.py</code> -> Functions which use the trained neural networks to recognize gestures from keypoints.</li>
<li> <code>gesture_executor.py</code> -> Functions which implement the end action with an input gesture. E.g. Left Click, Reduce Screen Brightness</li>
<li> <code>config.py</code> -> Stores the configuration and state of the application in dataclasses for easy access. </li>
<li> <code>user_config.py</code> -> Stores the definition of all the actions that will be executed when a particular gesture is detected. </li>
</ul>
</details>

### Useful Information

[Joints of the hand](https://en.wikipedia.org/wiki/Interphalangeal_joints_of_the_hand)

[HandCommander](https://www.deuxexsilicon.com/handcommander/)

[Video recorded with VokoScreenNG](https://github.com/vkohaupt/vokoscreenNG)
