'''
Various Utility Functions
'''

import math
from functools import partial
from pynput.keyboard import Listener, Key

def calc_polar(x,y):
    ''' Calculate the polar form of the Cartesian coordinates x and y. '''
    return (x**2 + y**2)**0.5, math.atan2(y, x)/math.pi

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
