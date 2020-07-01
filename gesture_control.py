'''
This script receives the hand keypoints detected by the mediapipe through
zmq/protobuf and then uses them to control the mouse through pyautogui.
'''

import subprocess
import json
import zmq
import pyautogui
import torch
import numpy as np
from pynput.keyboard import Controller, KeyCode
from model import GestureNet, ShrecNet
from proto import landmarkList_pb2

##################
# Mouse Tracking #
##################

def get_avg_pointer_loc(pointer_buffer):
    '''Gets average of previous 5 pointer locations'''
    x = [i[0] for i in pointer_buffer]
    y = [i[1] for i in pointer_buffer]
    return sum(x)/len(pointer_buffer), sum(y)/len(pointer_buffer)


def calc_pointer(landmarks, pointer_buffer, ITER_COUNT):
    ''' Uses the landmarks to calculate the location of the cursor on the screen. '''

    # The tip of the index pointer is the eighth landmark in the list
    index_pointer = landmarks[8]['x'], landmarks[8]['y'], landmarks[8]['z']

    # Screen resolution
    resolution = pyautogui.size().width, pyautogui.size().height
    scaled_pointer = resolution[0]*index_pointer[0], resolution[1]*index_pointer[1]

    pointer_buffer[ITER_COUNT%5] = scaled_pointer
    actual_pointer = get_avg_pointer_loc(pointer_buffer)

    return actual_pointer, pointer_buffer


def mouse_track(current_pointer, prev_pointer, FLAGS):
    '''
    Performs mouse actions depending on the FLAGS that have been set.
    prev_pointer is only modified if the mouse is up and we are not scrolling.
    '''

    threshold = 100

    # If mouse is down and movement below threshold, do not move the mouse
    if FLAGS['mousedown'] and (abs(current_pointer[0] - prev_pointer[0]) +
                               abs(current_pointer[1] - prev_pointer[1]) < threshold):
        return prev_pointer
    elif FLAGS['scroll']:
        amt_to_scroll = (current_pointer[1] - prev_pointer[1])/10
        pyautogui.scroll(amt_to_scroll)
        return prev_pointer
    else:
        pyautogui.moveTo(current_pointer[0], current_pointer[1], 0)
        return current_pointer


####################
# Config Detection #
####################

def format_landmark(landmark, hand, mode, dims):
    ''' A wrapper over formt_static_landmark and format_dynamic_landmark. '''
    if mode == 'mouse':
        return format_static_landmark(landmark, hand, dims[0])
    else:
        return format_dynamic_landmark(landmark, dims[1])

def format_static_landmark(landmark, hand, input_dim):
    '''
    Formats the input keypoints into the format expected by GestureNet.
    Refer make_vector in train_model.py for more details
    '''
    formatted_landmark = np.empty((input_dim))
    for i in range(4):
        formatted_landmark[3*i] = landmark[i+1]['x'] - landmark[i]['x']
        formatted_landmark[3*i+1] = landmark[i+1]['y'] - landmark[i]['y']
        formatted_landmark[3*i+2] = landmark[i+1]['z'] - landmark[i]['z']

    for i in range(3):
        formatted_landmark[3*i+12] = landmark[i+6]['x'] - landmark[i+5]['x']
        formatted_landmark[3*i+13] = landmark[i+6]['y'] - landmark[i+5]['y']
        formatted_landmark[3*i+14] = landmark[i+6]['z'] - landmark[i+5]['z']

        formatted_landmark[3*i+21] = landmark[i+10]['x'] - landmark[i+9]['x']
        formatted_landmark[3*i+22] = landmark[i+10]['y'] - landmark[i+9]['y']
        formatted_landmark[3*i+23] = landmark[i+10]['z'] - landmark[i+9]['z']

        formatted_landmark[3*i+30] = landmark[i+14]['x'] - landmark[i+13]['x']
        formatted_landmark[3*i+31] = landmark[i+14]['y'] - landmark[i+13]['y']
        formatted_landmark[3*i+32] = landmark[i+14]['z'] - landmark[i+13]['z']

        formatted_landmark[3*i+39] = landmark[i+18]['x'] - landmark[i+17]['x']
        formatted_landmark[3*i+40] = landmark[i+18]['y'] - landmark[i+17]['y']
        formatted_landmark[3*i+41] = landmark[i+18]['z'] - landmark[i+17]['z']

    formatted_landmark[48] = hand
    return formatted_landmark


def format_dynamic_landmark(landmark, input_dim):
    '''
    Formats the input keypoints into the format expected by ShrecNet.
    Refer format_mediapipe in dynamic_train_model.py for format details.
    '''
    formatted_landmark = np.zeros((input_dim))
    # Absolute
    for i in range(21):
        formatted_landmark[2*i] = landmark[i]['x']
        formatted_landmark[2*i+1] = landmark[i]['y']

    # Relative
    for i in range(4):
        # calculate L01, L12, L23, L34
        formatted_landmark[42+2*i] = formatted_landmark[2*i+2] - formatted_landmark[2*i] #L__X
        formatted_landmark[42+2*i+1] = formatted_landmark[2*i+3] - formatted_landmark[2*i+1] #L__Y

    for i in range(3):
        # calculate L56, L67, L78
        formatted_landmark[50+2*i] = formatted_landmark[2*i+12] - formatted_landmark[2*i+10]
        formatted_landmark[50+2*i+1] = formatted_landmark[2*i+13] - formatted_landmark[2*i+11]

        # calculate L910, L1011, L1112
        formatted_landmark[56+2*i] = formatted_landmark[2*i+20] - formatted_landmark[2*i+18]
        formatted_landmark[56+2*i+1] = formatted_landmark[2*i+21] - formatted_landmark[2*i+19]

        # calculate L1314, L1415, L1516
        formatted_landmark[62+2*i] = formatted_landmark[2*i+28] - formatted_landmark[2*i+26]
        formatted_landmark[62+2*i+1] = formatted_landmark[2*i+29] - formatted_landmark[2*i+27]

        # calculate L1718, L1819, L1920
        formatted_landmark[68+2*i] = formatted_landmark[2*i+36] - formatted_landmark[2*i+34]
        formatted_landmark[68+2*i+1] = formatted_landmark[2*i+37] - formatted_landmark[2*i+35]

    return formatted_landmark


def get_gesture(nets, mappings, landmarks, mode, keypoint_buffer, iter_count):
    ''' A wrapper over get_static_gesture and get_dynamic_gesture. '''
    if mode == 'mouse':
        return get_static_gesture(nets[0], mappings[0], landmarks,
                                  keypoint_buffer)
    else:
        return get_dynamic_gesture(nets[1], mappings[1], landmarks,
                                   keypoint_buffer, iter_count)


def get_static_gesture(net, mapping, landmarks, keypoint_buffer):
    '''
    Uses the neural net to classify the keypoints into a gesture.
    Also decides if a 'bad gesture' was performed.
    '''
    landmarks = torch.tensor(landmarks, dtype=torch.float32)
    out = net(landmarks)
    #print(dict(zip(mapping.values(), softmax(out.detach().numpy()))))
    gesture_dict = dict(zip(mapping.values(), out.detach().numpy()))
    # doubling the likelihood of the bad gesture to prevent misclassification
    gesture_dict['bad'] *= 2
    gesture = max(gesture_dict, key=gesture_dict.get)
    return gesture, keypoint_buffer


def get_dynamic_gesture(net, mapping, landmarks, keypoint_buffer, iter_count):
    '''
    Detects a dynamic gesture using ShrecNet. Takes in a sequence of
    `DYNAMIC_BUFFER_LENGTH` keypoints and classifies it.
    '''

    # Left rotate buffer and replace last element,
    # essentially removing oldest and placing latest in the last position
    keypoint_buffer = torch.roll(keypoint_buffer, -1, dims=0)
    keypoint_buffer[-1] = torch.tensor(landmarks)

    # if the first index is a zero array, we have just switched modes.
    # Will stop being true after buffer_len iterations.
    # Also predict only once in 30 frames.
    if (keypoint_buffer[0] == torch.zeros(len(keypoint_buffer[0]))).all() \
       or iter_count % 30 != 0:
        return 'bad', keypoint_buffer


    out = net(torch.unsqueeze(keypoint_buffer, axis=0))
    gesture_dict = dict(zip(mapping.values(), out[0].detach().numpy()))

    # print(gesture_dict)
    # print(max(gesture_dict, key=gesture_dict.get))
    gesture = max(gesture_dict, key=gesture_dict.get)

    return gesture, keypoint_buffer


#################
# Config Action #
#################

def config_action(config, flags, modes, config_buffer, iter_count):
    '''
    Given a configuration, decides what action to perform.
    '''
    if modes[0] == 'mouse':
        return config_static_action(config, flags, modes, config_buffer, iter_count)
    else:
        return config_dynamic_action(config, flags, modes, config_buffer, iter_count)

def config_static_action(config, flags, modes, config_buffer, iter_count):
    '''
    Given a configuration, decides what action to perform.
    Modifies an array of flags based on what buttons are clicked
    bad -> invalid gesture
    seven -> left mouse down
    four -> right mouse down
    eight -> double click
    spiderman -> scroll
    hitchhike -> mode switch
    '''
    valid = valid_config(config, config_buffer) #check if valid gesture
    config_buffer[iter_count%5] = config  #adding the new config to the buffer

    if config == 'bad' or not valid:
        pyautogui.mouseUp()
        flags['mousedown'] = False
        flags['scroll'] = False
    elif config == 'hitchhike':
        # Rotating list
        modes = modes[-1:] + modes[:-1]
    elif config == 'seven':
        pyautogui.mouseDown()
        flags['mousedown'] = True
        flags['scroll'] = False
    elif config in ['four', 'eight', 'spiderman']:
        pyautogui.mouseUp()
        flags['mousedown'] = False
        if config == 'four':
            pyautogui.rightClick()
        elif config == 'eight':
            pyautogui.doubleClick()
        else: #spiderman
            flags['scroll'] = True

    return flags, modes, config_buffer


def config_dynamic_action(config, flags, modes, config_buffer, iter_count):
    '''
    Given a configuration, decides what action to perform.
    Swipe Left/Right -> Switch workspaces (Requires xdotool)
    Rotation Clockwise/Counter Clockwise -> Increae/decrese volume
    Swipe Up/Down -> Increase/decrease brightness
    Tap -> Screenshot
    Grab -> Mode switch
    Pinch/Expand -> Zoom in/out
    '''
    valid = valid_config(config, config_buffer) #check if valid gesture
    config_buffer[iter_count%5] = config  #adding the new config to the buffer
    keyboard = Controller()

    print(config)
    if config in ['Shake', 'bad'] or not valid:
        pass
    elif config in ['Swipe Left', 'Swipe Right']:
        if config == 'Swipe Left':
            subprocess.run(['xdotool', 'set_desktop', '--relative', '--', '-1'], check=False)
        if config == 'Swipe Right':
            subprocess.run(['xdotool', 'set_desktop', '--relative', '--', '1'], check=False)
    elif config in ['Swipe Up', 'Swipe Down']: # HACK
        if config == 'Swipe Up':
            keyboard.press(KeyCode.from_vk(269025026))
            keyboard.release(KeyCode.from_vk(269025026))
        else:
            keyboard.press(KeyCode.from_vk(269025027))
            keyboard.release(KeyCode.from_vk(269025027))
    elif config in ['Rotation Clockwise', 'Rotation Counter Clockwise']: # HACK
        if config == 'Rotation Clockwise':
            keyboard.press(KeyCode.from_vk(269025043))
            keyboard.release(KeyCode.from_vk(269025043))
        else:
            keyboard.press(KeyCode.from_vk(269025041))
            keyboard.release(KeyCode.from_vk(269025041))
    elif config in ['Pinch', 'Expand']:
        pass
    elif config == 'Tap':
        pass
    elif config == 'Grab':
        # empty keypoint buffer before switching modes
        pass
    else:
        pass

    return flags, modes, config_buffer

def valid_config(config, config_buffer):
    '''
    Checks whether the gesture performed is in the config buffer i.e. recently performed.
    For most gestures, if it is present, then that makes it an invalid gesture.
    This is to prevent multiple gesture detections in a short span of time.
    '''
    if config in ['bad', 'seven', 'spiderman', 'Shake']: # these gestures are always valid, even if repeated
        return True
    if config in config_buffer: # repeated gesture
        return False
    return True

########
# Main #
########

def main():
    ''' The main function '''

    # allow mouse to move to edge of screen, and set interval between calls to 0.01
    pyautogui.FAILSAFE = False
    pyautogui.PAUSE = 0.01

    # Setting up connection
    context = zmq.Context()
    sock = context.socket(zmq.PULL)
    sock.connect("tcp://127.0.0.1:5556")

    landmarkList = landmarkList_pb2.LandmarkList()

    # maintain a buffer of most recent movements to smoothen mouse movement
    pointer_buffer = [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0)]
    prev_pointer = 0, 0
    # maintain a buffer of most recently detected configs
    config_buffer = ['', '', '', '', '']
    ITER_COUNT = 0

    # array of flags for mouse control
    FLAGS = {'mousedown': False, 'scroll': False}


    STATIC_INPUT_DIM = 49 #refer make_vector() in model.py to verify input dimensions
    STATIC_OUTPUT_CLASSES = 6
    STATIC_PATH = 'models/gesture_net'

    DYNAMIC_INPUT_DIM = 74
    DYNAMIC_OUTPUT_CLASSES = 14
    DYNAMIC_PATH = 'models/shrec_net'

    DYNAMIC_BUFFER_LENGTH = 60
    # maintain a buffer of most recently detected keypoints for dynamic gestures
    keypoint_buffer = torch.zeros((DYNAMIC_BUFFER_LENGTH, DYNAMIC_INPUT_DIM))

    # Setting up networks
    gesture_net = GestureNet(STATIC_INPUT_DIM, STATIC_OUTPUT_CLASSES)
    gesture_net.load_state_dict(torch.load(STATIC_PATH))
    gesture_net.eval()

    shrec_net = ShrecNet(DYNAMIC_INPUT_DIM, DYNAMIC_OUTPUT_CLASSES)
    shrec_net.load_state_dict(torch.load(DYNAMIC_PATH))
    shrec_net.eval()

    # Modes:
    # Each mode is a different method of interaction with the system.
    # Any given functionality might require the use of multiple modes
    # The first index represents the current mode. When mode is switched, the list is cycled.
    # 1. Mouse -> In mouse mode, we interact with the system in all the ways that a mouse can.
    #             E.g. left click, right click, scroll
    # 2. Gesture -> Intuitive gesture are performed to do complicated actions, such as switch
    #             worskpace, dim screen brightness etc.
    MODES = ['mouse', 'gesture']

    # Fetching gesture mapping
    with open('data/gesture_mapping.json', 'r') as jsonfile:
        static_gesture_mapping = json.load(jsonfile)
    with open('data/dynamic_gesture_mapping.json', 'r') as jsonfile:
        dynamic_gesture_mapping = json.load(jsonfile)

    # Modules:
    # Mouse Tracking - responsible for tracking and moving the cursor
    # Config Detection - takes in the keypoints and outputs a configuration
    # Config Action - takes in a configuration and maps it to an action i.e. LeftClick

    while True:
        data = sock.recv()
        landmarkList.ParseFromString(data)
        landmarks = []
        for lmark in landmarkList.landmark:
            landmarks.append({'x': lmark.x, 'y': lmark.y, 'z': lmark.z})

        # Handedness - true if right hand, false if left
        handedness = landmarkList.handedness

        # The mouse tracking module is only called if the current mode is 'mouse'
        if MODES[0] == 'mouse':
            ##################
            # Mouse Tracking #
            ##################

            # get pointer location
            mouse_pointer, pointer_buffer = calc_pointer(landmarks, pointer_buffer, ITER_COUNT)

            # control the mouse
            prev_pointer = mouse_track(mouse_pointer, prev_pointer, FLAGS)

        ####################
        # Config Detection #
        ####################

        input_data = format_landmark(landmarks, handedness, MODES[0],
                                    (STATIC_INPUT_DIM, DYNAMIC_INPUT_DIM))
        GESTURE, keypoint_buffer = get_gesture((gesture_net, shrec_net),
                                               (static_gesture_mapping, dynamic_gesture_mapping),
                                               input_data, MODES[0], keypoint_buffer, ITER_COUNT)

        #################
        # Config Action #
        #################

        FLAGS, MODES, config_buffer = config_action(GESTURE, FLAGS, MODES,
                                                    config_buffer, ITER_COUNT)


        ITER_COUNT += 1


if __name__ == '__main__':
    main()
