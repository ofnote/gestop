'''
Various Utility Functions
'''

import math
from functools import partial
import logging
import json
import os
from pynput.keyboard import Listener, Key
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import torch
from ..config import package_directory

def calc_polar(x,y):
    ''' Calculate the polar form of the Cartesian coordinates x and y. '''
    return (x**2 + y**2)**0.5, math.atan2(y, x)/math.pi

def on_press(S, key):
    ''' Tracks keypresses. Sets ctrl_flag if the ctrl key is pressed.'''
    if key == Key.ctrl:
        S.ctrl_flag = True

def on_release(S, key):
    ''' Tracks keypresses. Unsets ctrl_flag if the ctrl key is released.'''
    if key == Key.ctrl:
        S.ctrl_flag = False

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

def update_static_mapping(static_gesture_filepath):
    '''
    Fit the LabelEncoder on static gesture data and write mapping to disk
    '''
    data = pd.read_csv(static_gesture_filepath)
    data = data['GESTURE']
    le = LabelEncoder()
    le.fit(data)

    # Store mapping to disk
    le_name_mapping = dict(zip([int(i) for i in le.transform(le.classes_)], le.classes_))
    logging.info(le_name_mapping)
    with open(os.path.join(package_directory, "data/static_gesture_mapping.json"), 'w') as f:
        f.write(json.dumps(le_name_mapping))

    return le

def init_seed(seed):
    ''' Initializes random seeds for reproducibility '''
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
