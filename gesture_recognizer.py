'''
Functionality to recognize a static/dynamic gesture,
given a set of keypoints.
'''

import numpy as np
import torch

def format_landmark(landmark, hand, C, mode):
    ''' A wrapper over format_static_landmark and format_dynamic_landmark. '''
    if mode == 'mouse':
        return format_static_landmark(landmark, hand, C.static_input_dim)
    else:
        return format_dynamic_landmark(landmark, C.dynamic_input_dim)

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
    formatted_landmark[0] = landmark[0]['x']
    formatted_landmark[1] = landmark[0]['y']

    # Relative
    for i in range(4):
        # calculate L01, L12, L23, L34
        formatted_landmark[2+2*i] = landmark[i+1]['x'] - landmark[i]['x'] #L__X
        formatted_landmark[2+2*i+1] = landmark[i+1]['y'] - landmark[i]['y'] #L__Y

    for i in range(3):
        # calculate L56, L67, L78
        formatted_landmark[10+2*i] = landmark[i+6]['x'] - landmark[i+5]['x']
        formatted_landmark[10+2*i+1] = landmark[i+6]['y'] - landmark[i+5]['y']

        # calculate L910, L1011, L1112
        formatted_landmark[16+2*i] = landmark[i+10]['x'] - landmark[i+9]['x']
        formatted_landmark[16+2*i+1] = landmark[i+10]['y'] - landmark[i+9]['y']

        # calculate L1314, L1415, L1516
        formatted_landmark[22+2*i] = landmark[i+14]['x'] - landmark[i+13]['x']
        formatted_landmark[22+2*i+1] = landmark[i+14]['y'] - landmark[i+13]['y']

        # calculate L1718, L1819, L1920
        formatted_landmark[28+2*i] = landmark[i+18]['x'] - landmark[i+17]['x']
        formatted_landmark[28+2*i+1] = landmark[i+18]['y'] - landmark[i+17]['y']

    return formatted_landmark


def get_gesture(landmarks, C, S):
    ''' A wrapper over get_static_gesture and get_dynamic_gesture. '''
    if S.modes[0] == 'mouse':
        return get_static_gesture(landmarks, C), S
    else:
        return get_dynamic_gesture(landmarks, C, S)


def get_static_gesture(landmarks, C):
    '''
    Uses GestureNet to classify the keypoints into a gesture.
    Also decides if a 'bad gesture' was performed.
    '''
    landmarks = torch.tensor(landmarks, dtype=torch.float32)
    out = C.gesture_net(landmarks)
    #print(dict(zip(mapping.values(), softmax(out.detach().numpy()))))
    gesture_dict = dict(zip(C.static_gesture_mapping.values(), out.detach().numpy()))
    # doubling the likelihood of the bad gesture to lower chances of misclassification
    gesture_dict['bad'] *= 2
    gesture = max(gesture_dict, key=gesture_dict.get)
    return gesture


def get_dynamic_gesture(landmarks, C, S):
    '''
    Detects a dynamic gesture using ShrecNet. Takes in a sequence of
    `DYNAMIC_BUFFER_LENGTH` keypoints and classifies it.
    '''
    # Using the variable CTRL_FLAG to detect if the ctrl key is set.
    # Flag set by the Listener Thread (see gesture_receiver.py)
    # Detection only occurs when the Ctrl key is released.

    # |-------------------------------------------------------------------|
    # |CTRL_FLAG | PREV_FLAG | Comments                                   |
    # |----------+-----------+--------------------------------------------|
    # | False    | False     | Ctrl key has not been pressed.             |
    # | True     | False     | Just been pressed. Start storing keypoints.|
    # | True     | True      | Continuing capturing keypoints.            |
    # | False    | True      | Key has been released. Time to detect.     |
    # |----------+-----------+--------------------------------------------|

    if not S.ctrl_flag and not S.prev_flag:
        return 'bad', S

    # Store keypoints in buffer
    S.keypoint_buffer.append(torch.tensor(landmarks))

    gesture = 'bad'
    # Refer table above
    if not S.ctrl_flag and S.prev_flag:
        gesture, S = dynamic_gesture_detection(C, S)
        S.keypoint_buffer = []

    S.prev_flag = S.ctrl_flag
    return gesture, S


def dynamic_gesture_detection(C, S):
    ''' Detection of Dynamic Gesture using ShrecNet. '''

    # Formatting network input
    x = torch.unsqueeze(torch.stack(S.keypoint_buffer), 0)
    out = C.shrec_net(x.float())
    gesture_dict = dict(zip(C.dynamic_gesture_mapping.values(), out[0].detach().numpy()))

    gesture = max(gesture_dict, key=gesture_dict.get)

    return gesture, S
