'''
Functionality to recognize a static/dynamic gesture,
given a set of keypoints.
'''
import numpy as np
import torch
from .util.utils import calc_polar

# Using the variable ctrl_flag to detect if the ctrl key is set.
# Flag set by the Listener Thread (see gesture_receiver.py)
# prev_flag is the value of ctrl_flag in the previous timestep.
# Dynamic gestures are only detected when the Ctrl key is released,
# i.e. ctrl_flag is True and prev_flag is False.

# |----------------------------------------------------------------------------------------|
# |CTRL_FLAG | PREV_FLAG | Comments                                                        |
# |----------+-----------+-----------------------------------------------------------------|
# | False    | False     | Ctrl key has not been pressed. Detect static gesture.           |
# | True     | False     | Just been pressed. Start storing keypoints for dynamic gesture. |
# | True     | True      | Continuing capturing keypoints for dynamic gesture.             |
# | False    | True      | Key has been released. Time to detect dynamic gesture.          |
# |----------+-----------+-----------------------------------------------------------------|


def format_landmark(landmark, hand, C, S):
    ''' A wrapper over format_static_landmark and format_dynamic_landmark. '''
    if S.ctrl_flag or S.prev_flag:
        return format_dynamic_landmark(landmark, C.dynamic_input_dim, S)
    else:
        return format_static_landmark(landmark, hand, C.static_input_dim)

def format_static_landmark(landmark, hand, input_dim):
    '''
    Formats the input keypoints into the format expected by StaticNet.
    Refer make_vector in static_train_model.py for more details
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


def format_dynamic_landmark(landmark, input_dim, S):
    '''
    Formats the input keypoints into the format expected by ShrecNet.
    Refer construct_seq in dynamic_train_model.py for format details.
    '''
    formatted_landmark = np.zeros((input_dim))
    # Absolute
    formatted_landmark[0] = landmark[0]['x']
    formatted_landmark[1] = landmark[0]['y']

    if S.prev_landmark is None: # start of sequence
        formatted_landmark[2] = 0
        formatted_landmark[3] = 0
    else:  # change in postiion in polar coordinates
        x = formatted_landmark[0] - S.prev_landmark[0]
        y = formatted_landmark[1] - S.prev_landmark[1]
        formatted_landmark[2], formatted_landmark[3] = calc_polar(x, y)

    # Relative
    for i in range(4):
        # calculate L01, L12, L23, L34
        x = landmark[i+1]['x'] - landmark[i]['x'] #L__X
        y = landmark[i+1]['y'] - landmark[i]['y'] #L__Y
        formatted_landmark[4+2*i], formatted_landmark[4+2*i+1] = x, y

    for i in range(3):
        # calculate L56, L67, L78
        x = landmark[i+6]['x'] - landmark[i+5]['x']
        y = landmark[i+6]['y'] - landmark[i+5]['y']
        formatted_landmark[12+2*i], formatted_landmark[12+2*i+1] = x, y

        # calculate L910, L1011, L1112
        x = landmark[i+10]['x'] - landmark[i+9]['x']
        y = landmark[i+10]['y'] - landmark[i+9]['y']
        formatted_landmark[18+2*i], formatted_landmark[18+2*i+1] = x, y

        # calculate L1314, L1415, L1516
        x = landmark[i+14]['x'] - landmark[i+13]['x']
        y = landmark[i+14]['y'] - landmark[i+13]['y']
        formatted_landmark[24+2*i], formatted_landmark[24+2*i+1] = x, y

        # calculate L1718, L1819, L1920
        x = landmark[i+18]['x'] - landmark[i+17]['x']
        y = landmark[i+18]['y'] - landmark[i+17]['y']
        formatted_landmark[30+2*i], formatted_landmark[30+2*i+1] = x, y

    S.prev_landmark = formatted_landmark

    return formatted_landmark


def get_gesture(landmarks, C, S):
    ''' A wrapper over get_static_gesture and get_dynamic_gesture. '''
    if S.ctrl_flag or S.prev_flag:
        return get_dynamic_gesture(landmarks, C, S)
    else:
        return get_static_gesture(landmarks, C), S


def get_static_gesture(landmarks, C):
    '''
    Uses StaticNet to classify the keypoints into a gesture.
    Also decides if a 'bad gesture' was performed.
    '''
    landmarks = torch.tensor(landmarks, dtype=torch.float32)
    out = C.static_net(landmarks)
    #print(dict(zip(mapping.values(), softmax(out.detach().numpy()))))
    gesture_dict = dict(zip(C.static_gesture_mapping.values(), out.detach().numpy()))
    # doubling the likelihood of the bad gesture to lower chances of misclassification
    gesture_dict['bad'] *= 2
    gesture = max(gesture_dict, key=gesture_dict.get)
    return gesture


def get_dynamic_gesture(landmarks, C, S):
    '''
    Detects a dynamic gesture using ShrecNet. Takes in a sequence of
    keypoints and classifies it.
    '''

    # Store keypoints in buffer
    S.keypoint_buffer.append(torch.tensor(landmarks))

    gesture = 'gesture_in_progress'

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
    out = C.dynamic_net(x.float())
    gesture_dict = dict(zip(C.dynamic_gesture_mapping.values(), out[0].detach().numpy()))

    gesture = max(gesture_dict, key=gesture_dict.get)

    # Reset prev_landmark for next detection
    S.prev_landmark = None

    return gesture, S
