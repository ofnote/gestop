'''
Receives the data from mediapipe
Interfaces with the gesture recognizer and the gesture executor modules.
'''

import json
import pyautogui
import torch
import zmq
from proto import landmarkList_pb2
from model import GestureNet, ShrecNet

from mouse_tracker import mouse_track, calc_pointer
from gesture_recognizer import format_landmark, get_gesture
from gesture_executor import config_action


def get_landmarks(data, landmark_list):
    ''' Parses the protobuf received from mediapipe and formats into a dict. '''
    landmark_list.ParseFromString(data)
    landmarks = []
    for lmark in landmark_list.landmark:
        landmark = ({'x': lmark.x, 'y': lmark.y, 'z': lmark.z})
        #print(landmark)
        landmarks.append(landmark)
    #print(landmarks)

    return landmarks, landmark_list.handedness

# class LandmarkHandler(socketserver.BaseRequestHandler):
#     """
#     The request handler class for our server.

#     It is instantiated once per connection to the server, and must
#     override the handle() method to implement communication to the
#     client.
#     """

#     def setup(self):
#         self.LandmarkList = landmarkList_pb2.LandmarkList()
#         self.C = initialize_gesture_recognizer()

#     def handle(self):
#         # self.request is the TCP socket connected to the client
#         print("{} wrote:".format(self.client_address[0]))
#         count_empty = 0

#         while True:
#             self.data = self.request.recv()
#             # Modules:
#             # Mouse Tracking - responsible for tracking and moving the cursor
#             # Config Detection - takes in the keypoints and outputs a configuration
#             # Config Action - takes in a configuration and maps it to an action i.e. LeftClick

#             # detect empty data (alias for client disconnected)
#             if self.data == b'':
#                 count_empty += 1
#             if count_empty > 100 : break

#             landmarks, handedness = get_landmarks(self.data, self.LandmarkList)
#             print(handedness)
#             # for l in landmarks: print(l)
#             print("No. of landmarks:", len(landmarks))
#             # get pointer location
#             mouse_pointer, self.C['pointer_buffer'] = calc_pointer(landmarks, self.C['pointer_buffer'], self.C['iter'])

#             # control the mouse
#             self.C['prev_pointer'] = mouse_track(mouse_pointer, self.C['prev_pointer'], self.C['flags'])
#             self.C['iter'] += 1
#             # mouse tracker
#             # run recognizer
#             # run action executor
#         # exit(0)


# def run_socket_server():
#     ''' Starts the server which receives data from medipipe. '''
#     HOST, PORT = "0.0.0.0", 8089

#     # Create the server
#     with socketserver.TCPServer((HOST, PORT), LandmarkHandler) as server:
#         print(f'Server now listening {PORT}')
#         server.serve_forever()
def initialize_gesture_recognizer():
    ''' Initialize the gesture recognizer for further use. '''

    # Dictionary holding all useful variables, parameters.
    C = {}

    # allow mouse to move to edge of screen, and set interval between calls to 0.01
    pyautogui.FAILSAFE = False
    pyautogui.PAUSE = 0.01

    # maintain a buffer of most recent movements to smoothen mouse movement
    C['pointer_buffer'] = [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0)]
    C['prev_pointer'] = 0, 0

    # maintain a buffer of most recently detected configs
    C['static_config_buffer'] = ['', '', '', '', '']
    C['dynamic_config_buffer'] = ['' for i in range(60)]
    #number of iterations
    C['iter'] = 0

    # array of flags for mouse control
    C['flags'] = {'mousedown': False, 'scroll': False}

    C['static_input_dim'] = 49 #refer make_vector() in model.py to verify input dimensions
    C['static_output_classes'] = 6

    C['dynamic_input_dim'] = 74
    C['dynamic_output_classes'] = 14


    C['dynamic_buffer_length'] = 60
    # maintain a buffer of most recently detected keypoints for dynamic gestures
    C['keypoint_buffer'] = torch.zeros((C['dynamic_buffer_length'], C['dynamic_input_dim']))

    # Modes:
    # Each mode is a different method of interaction with the system.
    # Any given functionality might require the use of multiple modes
    # The first index represents the current mode. When mode is switched, the list is cycled.
    # 1. Mouse -> In mouse mode, we interact with the system in all the ways that a mouse can.
    #             E.g. left click, right click, scroll
    # 2. Gesture -> Intuitive gesture are performed to do complicated actions, such as switch
    #             worskpace, dim screen brightness etc.
    C['modes'] = ['mouse', 'gesture']

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

def handle_zmq_stream():
    ''' Handles the incoming stream of data from mediapipe. '''
    C = initialize_gesture_recognizer()

    # setup zmq context
    context = zmq.Context()
    sock = context.socket(zmq.PULL)
    sock.connect("tcp://127.0.0.1:5556")

    landmark_list = landmarkList_pb2.LandmarkList()

    # Main while loop
    while True:
        data = sock.recv()

        landmarks, handedness = get_landmarks(data, landmark_list)
        mode = C['modes'][0] #current mode

        if mode == 'mouse':
            ##################
            # Mouse Tracking #
            ##################

            # get pointer location
            mouse_pointer, C = calc_pointer(landmarks, C)
            # control the mouse
            C = mouse_track(mouse_pointer, C)

        ####################
        # Config Detection #
        ####################

        input_data = format_landmark(landmarks, handedness, C)
        gesture, C = get_gesture(input_data, C)

        #################
        # Config Action #
        #################

        C = config_action(gesture, C)
        if mode != C['modes'][0]:  #mode switch
            #emoty buffer
            C['keypoint_buffer'] = torch.zeros((C['dynamic_buffer_length'], C['dynamic_input_dim']))

        C['iter'] += 1


if __name__ == "__main__":
    # run_socket_server()
    handle_zmq_stream()
