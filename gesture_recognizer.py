'''
Functions which implement code to track the mouse,
given a set of landmarks.
'''

import numpy as np
import torch

def format_landmark(landmark, hand, C):
    ''' A wrapper over formt_static_landmark and format_dynamic_landmark. '''
    if C['modes'][0] == 'mouse':
        return format_static_landmark(landmark, hand, C['static_input_dim'])
    else:
        return format_dynamic_landmark(landmark, C['dynamic_input_dim'])

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
    for i in range(21):
        formatted_landmark[2*i] = landmark[i]['x']
        formatted_landmark[2*i+1] = landmark[i]['y']

    # Relative
    for i in range(4):
        # calculate L01, L12, L23, L34
        formatted_landmark[42+2*i] = formatted_landmark[2*i+2] - formatted_landmark[2*i] #L__X
        formatted_landmark[42+2*i+1] = formatted_landmark[2*i+3] - formatted_landmark[2*i+1] #L__Y

    for i in range(3):
        # calculate L56, L67, L78
        formatted_landmark[50+2*i] = formatted_landmark[2*i+12] - formatted_landmark[2*i+10]
        formatted_landmark[50+2*i+1] = formatted_landmark[2*i+13] - formatted_landmark[2*i+11]

        # calculate L910, L1011, L1112
        formatted_landmark[56+2*i] = formatted_landmark[2*i+20] - formatted_landmark[2*i+18]
        formatted_landmark[56+2*i+1] = formatted_landmark[2*i+21] - formatted_landmark[2*i+19]

        # calculate L1314, L1415, L1516
        formatted_landmark[62+2*i] = formatted_landmark[2*i+28] - formatted_landmark[2*i+26]
        formatted_landmark[62+2*i+1] = formatted_landmark[2*i+29] - formatted_landmark[2*i+27]

        # calculate L1718, L1819, L1920
        formatted_landmark[68+2*i] = formatted_landmark[2*i+36] - formatted_landmark[2*i+34]
        formatted_landmark[68+2*i+1] = formatted_landmark[2*i+37] - formatted_landmark[2*i+35]

    return formatted_landmark


def get_gesture(landmarks, C):
    ''' A wrapper over get_static_gesture and get_dynamic_gesture. '''
    if C['modes'][0] == 'mouse':
        return get_static_gesture(landmarks, C)
    else:
        return get_dynamic_gesture(landmarks, C)


def get_static_gesture(landmarks, C):
    '''
    Uses the neural net to classify the keypoints into a gesture.
    Also decides if a 'bad gesture' was performed.
    '''
    landmarks = torch.tensor(landmarks, dtype=torch.float32)
    out = C['gesture_net'](landmarks)
    #print(dict(zip(mapping.values(), softmax(out.detach().numpy()))))
    gesture_dict = dict(zip(C['static_gesture_mapping'].values(), out.detach().numpy()))
    # doubling the likelihood of the bad gesture to prevent misclassification
    gesture_dict['bad'] *= 2
    gesture = max(gesture_dict, key=gesture_dict.get)
    return gesture, C


def get_dynamic_gesture(landmarks, C):
    '''
    Detects a dynamic gesture using ShrecNet. Takes in a sequence of
    `DYNAMIC_BUFFER_LENGTH` keypoints and classifies it.
    '''

    # Left rotate buffer and replace last element,
    # essentially removing oldest and placing latest in the last position
    C['keypoint_buffer'] = torch.roll(C['keypoint_buffer'], -1, dims=0)
    C['keypoint_buffer'][-1] = torch.tensor(landmarks)

    # if the first index is a zero array, we have just switched modes.
    # Will stop being true after buffer_len iterations.
    # Also predict only once in 30 frames.
    if (C['keypoint_buffer'][0] == torch.zeros(len(C['keypoint_buffer'][0]))).all() \
       or C['iter'] % 20 != 0:
        return 'bad', C


    out = C['shrec_net'](torch.unsqueeze(C['keypoint_buffer'], axis=0))
    gesture_dict = dict(zip(C['dynamic_gesture_mapping'].values(), out[0].detach().numpy()))

    # print(gesture_dict)
    # print(max(gesture_dict, key=gesture_dict.get))
    gesture = max(gesture_dict, key=gesture_dict.get)

    return gesture, C
