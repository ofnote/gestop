'''
This script receives the hand keypoints detected by the mediapipe through
zmq/protobuf and then uses them to control the mouse through pyautogui.
'''

import math
import zmq
import pyautogui
import torch
import numpy as np
from model import GestureNet
from proto import landmarkList_pb2


##################
# Mouse Tracking #
##################

def get_avg_pointer_loc(pointer_buffer):
    '''Gets average of previous 5 pointer locations'''
    x = [i[0] for i in pointer_buffer]
    y = [i[1] for i in pointer_buffer]
    return sum(x)/len(pointer_buffer), sum(y)/len(pointer_buffer)

####################
# Config Detection #
####################

    ########################
    # Rule-Based Detection #
    ########################

def calculate_angles(landmarks):
    '''
    Calculates the various angles from the joints of the hand,
    see `Useful Information` in README.md for more details
    PIP (Proximal InterPhalangeal) angles - lower joint angles (5)
    DIP (Dostal InterPhalangeal) angles - upper joint angles (5)
    MCP (MetaCarpoPhalangeal) angles - angles between fingers (4)
    Palm angle - rotation of the hand with respect to the vertical axis (1)
    '''
    angles = {}
    angles['dip'] = []
    angles['pip'] = []
    angles['mcp'] = []
    for i in range(5):
        angles['dip'].append(angle_between_lines(landmarks[3+(4*i)]['x'], landmarks[3+(4*i)]['y'],
                                                 landmarks[3+(4*i)+1]['x'], landmarks[3+(4*i)+1]['y'],
                                                 landmarks[3+(4*i)-1]['x'], landmarks[3+(4*i)-1]['y'],
                                                 ))  # L3,L7,L11,L15,L19
        angles['pip'].append(angle_between_lines(landmarks[2+(4*i)]['x'], landmarks[2+(4*i)]['y'],
                                                 landmarks[2+(4*i)+1]['x'], landmarks[2+(4*i)+1]['y'],
                                                 landmarks[2+(4*i)-1]['x'], landmarks[2+(4*i)-1]['y'],
                                                 ))  # L2,L6,L10,L14,L18


    for i in range(4):
        angles['mcp'].append(angle_between_lines(landmarks[1+(4*i)]['x'], landmarks[1+(4*i)]['y'],
                                                 landmarks[1+(4*i)+3]['x'], landmarks[1+(4*i)+3]['y'],
                                                 landmarks[1+(4*i)+7]['x'], landmarks[1+(4*i)+7]['y'],
                                                 ))  # L1,L5,L9,L13

    angles['palm'] = angle_between_lines(landmarks[0]['x'], landmarks[0]['y'],
                                         landmarks[9]['x'], landmarks[9]['y'],
                                         0, landmarks[0]['y'])  # L2,L6,L10,L14,L18
    return angles


def angle_between_lines(x0, y0, x1, y1, x2, y2):
    ''' Calculate the angle that (x1,y1) and (x2,y2) make at (x0,y0) '''
    angle1 = math.degrees(math.atan2(y0-y1, x0-x1))
    angle2 = math.degrees(math.atan2(y0-y2, x0-x2))

    return angle1 - angle2

def get_configuration(angles):
    ''' Using the calculated angles, outputs a high level configuration of the fingers. '''
    handState = []
    if angles['pip'][0] + angles['dip'][0] < 400:  # thumbAngle
        handState.append('straight')
    else:
        handState.append('bent')

    for i in range(1, 5):
        if angles['pip'][i] + angles['dip'][i] > 0:
            handState.append('straight')
        else:
            handState.append('bent')

    return handState

    ##############################
    # Neural Net Based Detection #
    ##############################

def format_landmark(landmark, hand, input_dim):
    '''
    Formats the input keypoints into the format expected by the neural net.
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

def get_gesture(net, landmarks):
    '''
    Uses the neural net to classify the keypoints into a gesture.
    Also decides if a 'bad gesture' was performed.
    '''
    gesture = 'bad'
    landmarks = torch.tensor(landmarks, dtype=torch.float32)
    out = net(landmarks)
    print(out)
    return gesture



#################
# Config Action #
#################

def config_action(config):
    ''' Given a configuration, decides what action to perform. '''
    if(config == ['straight', 'straight', 'bent', 'bent', 'bent']):
        pyautogui.mouseDown()
        return True
    else:
        pyautogui.mouseUp()
        return False


########
# Main #
########

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
ITER_COUNT = 0
MOUSEDOWN_FLAG = False
THRESHOLD = 100
OUTPUT_CLASSES = 4
INPUT_DIM = 49 #refer make_vector() in model.py to verify input dimensions
PATH = 'models/gesture_net'

gesture_net = GestureNet(INPUT_DIM, OUTPUT_CLASSES)
gesture_net.load_state_dict(torch.load(PATH))
gesture_net.eval()

# Modules:
# Mouse Tracking - responsible for tracking and moving the cursor
# Config Detection - takes in the keypoints and outputs a configuration
# i.e. ['straight','straight', 'bent', 'bent', 'bent'] (subject to change)
# Config Action - takes in a configuration and maps it to an action i.e. LeftClick

while True:
    data = sock.recv()
    landmarkList.ParseFromString(data)
    landmarks = []
    for lmark in landmarkList.landmark:
        landmarks.append({'x': lmark.x, 'y': lmark.y, 'z': lmark.z})

    # Handedness - true if right hand, false if left
    handedness = landmarkList.handedness

    ##################
    # Mouse Tracking #
    ##################

    # The tip of the index pointer is the eighth landmark in the list
    index_pointer = landmarks[8]['x'], landmarks[8]['y'], landmarks[8]['z']

    # Screen resolution
    resolution = pyautogui.size().width, pyautogui.size().height
    scaled_pointer = resolution[0]*index_pointer[0], resolution[1]*index_pointer[1]

    pointer_buffer[ITER_COUNT%5] = scaled_pointer
    actual_pointer = get_avg_pointer_loc(pointer_buffer)

    # if mouse is down and movement below threshold, do not move the mouse
    if MOUSEDOWN_FLAG and (abs(actual_pointer[0] - prev_pointer[0]) +
                           abs(actual_pointer[0] - prev_pointer[0]) < THRESHOLD):
        pass
    else:
        pyautogui.moveTo(actual_pointer[0], actual_pointer[1], 0)
        prev_pointer = actual_pointer

    ####################
    # Config Detection #
    ####################

    #angles = calculate_angles(landmarks)
    #gesture = get_configuration(angles)
    #print(gesture)
    input_data = format_landmark(landmarks, handedness, INPUT_DIM)
    GESTURE = get_gesture(gesture_net, input_data)

    #################
    # Config Action #
    #################

    #MOUSEDOWN_FLAG = config_action(GESTURE)

    ITER_COUNT += 1
