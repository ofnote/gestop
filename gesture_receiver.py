'''
Receives the data from mediapipe
Interfaces with the gesture recognizer and the gesture executor modules.
'''

import torch
import zmq
from proto import landmarkList_pb2

from config import initialize_gesture_recognizer, initialize_state
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

    #################
    # Config Action #
    #################

    S = config_action(gesture, S)
    if mode != S['modes'][0]:  #mode switch
        # empty the buffer
        S['keypoint_buffer'] = torch.zeros((C['dynamic_buffer_length'], C['dynamic_input_dim']))

    return S

def handle_zmq_stream():
    ''' Handles the incoming stream of data from mediapipe. '''
    C = initialize_gesture_recognizer()

    # setup zmq context
    context = zmq.Context()
    sock = context.socket(zmq.PULL)
    sock.connect("tcp://127.0.0.1:5556")

    landmark_list = landmarkList_pb2.LandmarkList()

    S = initialize_state(C)
    # Main while loop
    while True:
        data = sock.recv()

        landmarks, handedness = get_landmarks(data, landmark_list)

        S = handle_and_recognize(landmarks, handedness, C, S)

        S['iter'] += 1


if __name__ == "__main__":
    # run_socket_server()
    handle_zmq_stream()
