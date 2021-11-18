'''
Receives the data from mediapipe.
Interfaces with the gesture recognizer and the gesture executor modules.
'''

import argparse
import logging
import os
import socket

from .proto import landmarkList_pb2
from .config import Config, State, package_directory
from .mouse_tracker import mouse_track, calc_pointer
from .util.utils import on_press, on_release, start_key_listener
from .recognizer import format_landmark, get_gesture
from .executor import pose_action


def get_landmarks(data, landmark_list):
    ''' Parses the protobuf received from mediapipe and formats into a dict. '''
    landmark_list.ParseFromString(data)
    landmarks = []
    for lmark in landmark_list.landmark:
        landmark = ({'x': lmark.x, 'y': lmark.y, 'z': lmark.z})
        landmarks.append(landmark)

    return landmarks, landmark_list.handedness


def handle_and_recognize(landmarks, handedness, C, S):
    '''
    Given the keypoints from mediapipe:
    1. The mouse is tracked.
    2. A gesture is recognized, either static or dynamic
    3. The action corresponding to that gesture is executed.
    '''

    # For mouse tracking to occur, mouse tracking should be enabled
    # and the Ctrl key must not be pressed.
    if S.mouse_track and not S.ctrl_flag:
        ##################
        # Mouse Tracking #
        ##################

        # get pointer location
        mouse_pointer, S = calc_pointer(landmarks, S, C.resolution, C.map_coord)
        # track the mouse
        S = mouse_track(mouse_pointer, S, C.mouse, C.scroll_unit)

    ####################
    # Pose Detection #
    ####################

    input_data = format_landmark(landmarks, handedness, C, S)
    gesture, S = get_gesture(input_data, C, S)

    #################
    # Pose Action #
    #################

    if S.exec_action:
        S = pose_action(gesture, S, C)

    if gesture not in ['bad']:
        logging.info(f'handle_and_recognize: {gesture}:{S.action}')
    return S


def all_init(args):
    ''' Initializing the state and the configuration. '''

    C = Config(lite=False, config_path_json=args.config_path_json,
               config_path_py=args.config_path_py)
    S = State(mouse_track=args.mouse_track, exec_action=args.exec_action)

    start_key_listener(S)
    landmark_list = landmarkList_pb2.LandmarkList()

    return C, S, landmark_list

def process_data(data, landmark_list, C, S):
    landmarks, handedness = get_landmarks(data, landmark_list)

    if landmarks == []:
        raise ValueError("Landmarks not received")

    S = handle_and_recognize(landmarks, handedness, C, S)

def handle_stream(args):
    ''' Handles the incoming stream of data from mediapipe. '''

    C, S, landmark_list = all_init(args)

    # setup socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    HOST = '127.0.0.1'
    PORT = 5556
    sock.bind((HOST, PORT))
    sock.listen(1)

    logging.info('Receiver is ready!')
    try:
        while True: # Run server
            conn, addr = sock.accept()
            while True: # While connection is open
                data = conn.recv(4096)
                try:
                    process_data(data, landmark_list, C, S)
                except ValueError as v:
                    print("Closing Connection: ", v)
                    break
            conn.close()
        sock.close()
    except KeyboardInterrupt:
        print("\nGracefully shutting down")

if __name__ == "__main__":
    # Program runs on two threads
    # 1. Key Listener Thread -> Listens to what keys are being pressed
    # Dynamic gestures are only recognized if the Ctrl key is pressed
    # 2. MainThread -> The 'main' thread of execution
    # Receives, recognizes and executes gestures

    parser = argparse.ArgumentParser(description='An application to control the \
    desktop through hand gestures.')
    parser.add_argument("--no-mouse-track", help="Do not track mouse on startup",
                        dest="mouse_track", action='store_false')
    parser.add_argument("--config-path-json", help="Path to custom json configuration file",
                        type=str, default=os.path.join(package_directory, "data/action_config.json"))
    parser.add_argument("--config-path-py", help="Path to custom python configuration file",
                        type=str, default=os.path.join(package_directory, "user_config.py"))
    parser.add_argument("--no-action", help="Disbaled execution of actions. Useful for debugging.",
                        dest="exec_action", action='store_false')
    args = parser.parse_args()

    handle_stream(args)
    logging.info("Shutdown successfully")
