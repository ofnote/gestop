'''
Functions which initialize the 'configuration' -> values which don't change during runtime
and the 'state' -> valuse which represent the state of the application
'''
import json
import pyautogui
import torch
from model import GestureNet, ShrecNet

def initialize_gesture_recognizer():
    ''' Initialize the gesture recognizer for further use. '''

    # Dictionary holding all useful variables, parameters.
    C = {}

    # allow mouse to move to edge of screen, and set interval between calls to 0.01
    pyautogui.FAILSAFE = False
    pyautogui.PAUSE = 0.01

    C['static_input_dim'] = 49 #refer make_vector() in model.py to verify input dimensions
    C['static_output_classes'] = 6

    C['dynamic_input_dim'] = 74
    C['dynamic_output_classes'] = 14

    C['dynamic_buffer_length'] = 60

    # Fetching gesture mapping
    with open('data/gesture_mapping.json', 'r') as jsonfile:
        C['static_gesture_mapping'] = json.load(jsonfile)
    with open('data/dynamic_gesture_mapping.json', 'r') as jsonfile:
        C['dynamic_gesture_mapping'] = json.load(jsonfile)

    static_path = 'models/gesture_net'
    dynamic_path = 'models/shrec_net'
    # Setting up networks

    C['gesture_net'] = GestureNet(C['static_input_dim'], C['static_output_classes'])
    C['gesture_net'].load_state_dict(torch.load(static_path))
    C['gesture_net'].eval()

    C['shrec_net'] = ShrecNet(C['dynamic_input_dim'], C['dynamic_output_classes'])
    C['shrec_net'].load_state_dict(torch.load(dynamic_path))
    C['shrec_net'].eval()

    return C

def initialize_state(C):
    '''
    Initializes a set of parameters which will be modified while the application is runnning
    Represents the 'state' of the application
    '''
    S = {}

    # maintain a buffer of most recent movements to smoothen mouse movement
    S['pointer_buffer'] = [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0)]
    S['prev_pointer'] = 0, 0

    # Modes:
    # Each mode is a different method of interaction with the system.
    # Any given functionality might require the use of multiple modes
    # The first index represents the current mode. When mode is switched, the list is cycled.
    # 1. Mouse -> In mouse mode, we interact with the system in all the ways that a mouse can.
    #             E.g. left click, right click, scroll
    # 2. Gesture -> Intuitive gesture are performed to do complicated actions, such as switch
    #             worskpace, dim screen brightness etc.
    S['modes'] = ['mouse', 'gesture']
    # S['modes'] = ['gesture', 'mouse']

    # maintain a buffer of most recently detected configs
    S['static_config_buffer'] = ['', '', '', '', '']
    S['dynamic_config_buffer'] = ['' for i in range(30)]

    #number of iterations
    S['iter'] = 0

    # array of flags for mouse control
    S['flags'] = {'mousedown': False, 'scroll': False}

    # maintain a buffer of most recently detected keypoints for dynamic gestures
    S['keypoint_buffer'] = torch.zeros((C['dynamic_buffer_length'], C['dynamic_input_dim']))

    return S
