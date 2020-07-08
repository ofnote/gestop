'''
Functions which initialize the 'configuration' -> values which don't change during runtime
and the 'state' -> values which represent the state of the application
'''
import json
import pyautogui
import torch
from model import GestureNet, ShrecNet

def initialize_configuration():
    ''' Initialize the configuration of the gesture recognizer. '''

    # Dictionary holding all useful variables, parameters.
    C = {}

    # allow mouse to move to edge of screen, and set interval between calls to 0.01
    pyautogui.FAILSAFE = False
    pyautogui.PAUSE = 0.01

    # Seed value for reproducibility
    C['seed_val'] = 42

    # Refer make_vector() in train_model.py to verify input dimensions
    C['static_input_dim'] = 49
    C['static_output_classes'] = 6

    # Refer format_mediapipe() in dynamic_train_model.py to verify input dimensions
    C['dynamic_input_dim'] = 34
    C['dynamic_output_classes'] = 14

    # Fetching gesture mappings
    with open('data/gesture_mapping.json', 'r') as jsonfile:
        C['static_gesture_mapping'] = json.load(jsonfile)
    with open('data/dynamic_gesture_mapping.json', 'r') as jsonfile:
        C['dynamic_gesture_mapping'] = json.load(jsonfile)

    static_path = 'models/gesture_net'
    dynamic_path = 'models/shrec_net'

    if not torch.cuda.is_available():
        map_location = torch.device('cpu')
    else:
        map_location = None

    # Setting up networks

    print('Loading GestureNet...')
    C['gesture_net'] = GestureNet(C['static_input_dim'], C['static_output_classes'])
    C['gesture_net'].load_state_dict(torch.load(static_path, map_location=map_location))
    C['gesture_net'].eval()

    print('Loading ShrecNet..')

    C['shrec_net'] = ShrecNet(C['dynamic_input_dim'], C['dynamic_output_classes'])
    C['shrec_net'].load_state_dict(torch.load(dynamic_path, map_location=map_location))
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

    # maintain a buffer of keypoints for dynamic gestures
    S['keypoint_buffer'] = []

    # Flag to denote whether the Ctrl key is pressed
    S['CTRL_FLAG'] = False
    # CTRL_FLAG of the previous timestep. Used to detect change
    S['PREV_FLAG'] = False

    return S
