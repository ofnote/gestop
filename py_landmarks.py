'''
This script receives the hand keypoints detected by the mediapipe through
zmq/protobuf and then uses them to control the mouse through pyautogui.
'''

import math
import zmq
import pyautogui
from proto import landmarkList_pb2


# Calculate the angle that (x1,y1) and (x2,y2) make at (x0,y0)
def angle_between_lines(x0, y0, x1, y1, x2, y2):
    angle1 = math.degrees(math.atan2(y0-y1, x0-x1))
    angle2 = math.degrees(math.atan2(y0-y2, x0-x2))

    return angle1 - angle2


'''
 calculates the various angles from the joints of the hand,
 see `Useful Information` in README.md for more details
 PIP (Proximal InterPhalangeal) angles - lower joint angles (5)
 DIP (Dostal InterPhalangeal) angles - upper joint angles (5)
 MCP (MetaCarpoPhalangeal) angles - angles between fingers (4)
 Palm angle - rotation of the hand with respect to the vertical axis (1)
'''
def calculate_angles(landmarks):
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

# Gets average of previous 5 pointer locations
def get_avg_pointer_loc(pointer_buffer):
    x = [i[0] for i in pointer_buffer]
    y = [i[1] for i in pointer_buffer]
    return sum(x)/len(pointer_buffer), sum(y)/len(pointer_buffer)

# Using the calculated angles, outputs a high level configuration of the fingers
def get_configuration(angles):
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

# Given a configuration, decides what action to perform.
def config_action(config):
    if(config == ['straight', 'straight', 'bent', 'bent', 'bent']):
        pyautogui.mouseDown()
        return True
    else:
        pyautogui.mouseUp()
        return False

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

    angles = calculate_angles(landmarks)
    handState = get_configuration(angles)
    #print(fingerState)

    #################
    # Config Action #
    #################

    MOUSEDOWN_FLAG = config_action(handState)

    ITER_COUNT += 1
