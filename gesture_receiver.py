'''
Receives the data from mediapipe
Interfaces with the gesture recognizer and the gesture executor modules.
'''

import threading
from functools import partial
import argparse
import logging

import zmq
from pynput.keyboard import Listener, Key

from proto import landmarkList_pb2
from config import Config, State
from mouse_tracker import mouse_track, calc_pointer
from gesture_recognizer import format_landmark, get_gesture
from gesture_executor import config_action


def on_press(S, key):
    ''' Tracks keypresses. Sets ctrl_flag if the ctrl key is pressed.'''
    # print('{0} pressed'.format(key))
    if key == Key.ctrl:
        S.ctrl_flag = True

def on_release(S, key):
    ''' Tracks keypresses. Unsets ctrl_flag if the ctrl key is released.'''
    # print('{0} release'.format(key))
    if key == Key.ctrl:
        S.ctrl_flag = False
    if key == Key.esc:
        # Stop listener
        return False

def start_key_listener(S):
    ''' Starts the keypress listener. '''
    # Wrapping on_press and on_release into higher order functions
    # to avoid use of global variables
    on_press_key = partial(on_press, S)
    on_release_key = partial(on_release, S)

    # Start the listener on another thread to listen to keypress events
    listener = Listener(on_press=on_press_key,
                        on_release=on_release_key)
    listener.start()


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


def handle_and_recognize(landmarks, handedness, C, S):
    '''
    Given the keypoints from mediapipe:
    1. The mouse is tracked if the current mode is 'mouse'
    2. A gesture is recognized, either static or dynamic
    3. The action corresponding to that gesture is executed.
    '''
    mode = S.modes[0] #current mode

    if mode == 'mouse' and S.mouse_track:
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
    logging.info(f'handle_and_recognize: {gesture}')

    #################
    # Config Action #
    #################

    S = config_action(gesture, S, C)
    return S


def all_init(args):
    # Initializing the state and the configuration

    C = Config(lite=False, config_path=args.config_path)
    S = State(start_mode=args.start_mode, mouse_track=args.mouse_track)

    start_key_listener(S)
    landmark_list = landmarkList_pb2.LandmarkList()

    return C, S, landmark_list

def process_data(data, landmark_list, C, S):
    landmarks, handedness = get_landmarks(data, landmark_list)

    S = handle_and_recognize(landmarks, handedness, C, S)

def handle_zmq_stream(args):
    ''' Handles the incoming stream of data from mediapipe. '''

    C, S, landmark_list = all_init(args)

    # setup zmq context
    context = zmq.Context()
    sock = context.socket(zmq.PULL)
    sock.connect("tcp://127.0.0.1:5556")

    # Main while loop
    while True:
        data = sock.recv()
        process_data(data, landmark_list, C, S)

        # The key listener thread has shut down, leaving only MainThread
        if threading.active_count() == 1:
            break


if __name__ == "__main__":
    # run_socket_server()

    # Program runs on two threads
    # 1. Key Listener Thread -> Listens to what keys are being pressed
    # Dynamic gestures are only recognized if the Ctrl key is pressed
    # 2. MainThread -> The 'main' thread of execution
    # Receives, recognizes and executes gestures

    parser = argparse.ArgumentParser(description='An application to control the \
    desktop through hand gestures.')
    parser.add_argument("--no-mouse-track", help="Do not track mouse on startup",
                        dest="mouse_track", action='store_false')
    parser.add_argument("--start-mode", help="Mode to start the application in",
                        type=str, default="mouse", choices=['mouse', 'gesture'])
    parser.add_argument("--config-path", help="Path to custom configuration file",
                        type=str, default="data/action_config.json")

    args = parser.parse_args()

    handle_zmq_stream(args)
    logging.info("Shutdown successfully")
