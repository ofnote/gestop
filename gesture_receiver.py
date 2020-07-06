'''
Receives the data from mediapipe
Interfaces with the gesture recognizer and the gesture executor modules.
'''

import threading
from functools import partial

import torch
import zmq
from pynput.keyboard import Listener, Key

from proto import landmarkList_pb2
from config import initialize_configuration, initialize_state
from mouse_tracker import mouse_track, calc_pointer
from gesture_recognizer import format_landmark, get_gesture
from gesture_executor import config_action


def on_press(S, key):
    ''' Tracks keypresses. Sets the global CTRL_FLAG if the ctrl key is pressed.'''
    print('{0} pressed'.format(key))
    if key == Key.ctrl:
        S['CTRL_FLAG'] = True

def on_release(S, C, key):
    ''' Tracks keypresses. Unsets the global CTRL_FLAG if the ctrl key is released.'''
    print('{0} release'.format(key))
    if key == Key.ctrl:
        S['CTRL_FLAG'] = False
        # Empty keypoint buffer
        S['keypoint_buffer'] = torch.zeros((C['dynamic_buffer_length'], C['dynamic_input_dim']))
        S['buffer_len'] = 0
    if key == Key.esc:
        # Stop listener
        return False

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

def handle_and_recognize(landmarks, handedness, C, S):
    '''
    Given the keypoints from mediapipe
    The mouse is tracked if the current mode is 'mouse'
    A gesture is recognized, either static or dynamic
    And the action corresponding to that gesture is executed.
    '''
    mode = S['modes'][0] #current mode

    if mode == 'mouse':
        ##################
        # Mouse Tracking #
        ##################

        # get pointer location
        mouse_pointer, S = calc_pointer(landmarks, S)
        # control the mouse
        S = mouse_track(mouse_pointer, S)

    ####################
    # Config Detection #
    ####################

    input_data = format_landmark(landmarks, handedness, C, mode)
    gesture, S = get_gesture(input_data, C, S)
    # print(gesture)

    #################
    # Config Action #
    #################

    S = config_action(gesture, S)

    return S

def handle_zmq_stream():
    ''' Handles the incoming stream of data from mediapipe. '''

    # Initializing the state and the configuration
    C = initialize_configuration()
    S = initialize_state(C)

    # Wrapping on_press and on_release into higher order functions
    # to avoid use of global variables
    on_press_key = partial(on_press, S)
    on_release_key = partial(on_release, S, C)

    # Start the listener on another thread to listen to keypress events
    listener = Listener(on_press=on_press_key,
                        on_release=on_release_key)
    listener.start()

    # setup zmq context
    context = zmq.Context()
    sock = context.socket(zmq.PULL)
    sock.connect("tcp://127.0.0.1:5556")

    landmark_list = landmarkList_pb2.LandmarkList()

    # Main while loop
    while True:
        data = sock.recv()

        landmarks, handedness = get_landmarks(data, landmark_list)

        S = handle_and_recognize(landmarks, handedness, C, S)

        S['iter'] += 1

        # The key listener thread has shut down, leaving only GestureThread and MainThread
        if threading.active_count() == 1:
            break


if __name__ == "__main__":
    # run_socket_server()

    # Program runs on two threads
    # 1. Key Listener Thread -> Listens to what keys are being pressed
    # Dynamic gestures are only recognized if the Ctrl key is pressed
    # 2. MainThread -> The 'main' thread of execution
    # Receives, recognizes and executes gestures

    handle_zmq_stream()
    print("Shutdown successfully")
