'''
Functions which execute an action given a gesture config
'''
import subprocess
import pyautogui
from pynput.keyboard import Key, KeyCode, Controller

def config_action(config, C):
    '''
    Given a configuration, decides what action to perform.
    '''
    if C['modes'][0] == 'mouse':
        return config_static_action(config, C)
    else:
        return config_dynamic_action(config, C)

def config_static_action(config, C):
    '''
    Given a configuration, decides what action to perform.
    Modifies an array of flags based on what buttons are clicked
    bad -> invalid gesture
    seven -> left mouse down
    four -> right mouse down
    eight -> double click
    spiderman -> scroll
    hitchhike -> mode switch
    '''
    valid = valid_config(config, C['static_config_buffer']) #check if valid gesture
    C['static_config_buffer'][C['iter']%5] = config  #adding the new config to the buffer

    if config == 'bad' or not valid:
        pyautogui.mouseUp()
        C['flags']['mousedown'] = False
        C['flags']['scroll'] = False
    elif config == 'hitchhike':
        # Rotating list
        C['modes'] = C['modes'][-1:] + C['modes'][:-1]
    elif config == 'seven':
        pyautogui.mouseDown()
        C['flags']['mousedown'] = True
        C['flags']['scroll'] = False
    elif config in ['four', 'eight', 'spiderman']:
        pyautogui.mouseUp()
        C['flags']['mousedown'] = False
        if config == 'four':
            pyautogui.rightClick()
        elif config == 'eight':
            pyautogui.doubleClick()
        else: #spiderman
            C['flags']['scroll'] = True

    return C


def config_dynamic_action(config, C):
    '''
    Given a configuration, decides what action to perform.
    Swipe Left/Right -> Switch workspaces (Requires xdotool)
    Rotation Clockwise/Counter Clockwise -> Increae/decrese volume
    Swipe Up/Down -> Increase/decrease brightness
    Tap -> Screenshot
    Grab -> Mode switch
    Pinch/Expand -> Zoom in/out
    '''
    valid = valid_config(config, C['dynamic_config_buffer']) #check if valid gesture
    C['dynamic_config_buffer'][C['iter']%30] = config  #adding the new config to the buffer
    keyboard = Controller()

    if config in ['Shake', 'bad'] or not valid:
        pass
    elif config in ['Swipe Left', 'Swipe Right']:
        if config == 'Swipe Left':
            subprocess.run(['xdotool', 'set_desktop', '--relative', '--', '-1'], check=False)
        if config == 'Swipe Right':
            subprocess.run(['xdotool', 'set_desktop', '--relative', '--', '1'], check=False)
    elif config in ['Swipe Up', 'Swipe Down']: # HACK
        if config == 'Swipe Up':
            keyboard.press(KeyCode.from_vk(269025026))
            keyboard.release(KeyCode.from_vk(269025026))
        else:
            keyboard.press(KeyCode.from_vk(269025027))
            keyboard.release(KeyCode.from_vk(269025027))
    elif config in ['Rotation Clockwise', 'Rotation Counter Clockwise']: # HACK
        if config == 'Rotation Clockwise':
            keyboard.press(KeyCode.from_vk(269025043))
            keyboard.release(KeyCode.from_vk(269025043))
        else:
            keyboard.press(KeyCode.from_vk(269025041))
            keyboard.release(KeyCode.from_vk(269025041))
    elif config in ['Pinch', 'Expand']:
        pass
    elif config == 'Tap':
        keyboard.press(Key.print_screen)
        keyboard.release(Key.print_screen)
    elif config == 'Grab':
        # Rotating list to switch modes
        C['modes'] = C['modes'][-1:] + C['modes'][:-1]
    else:
        pass

    return C

def valid_config(config, config_buffer):
    '''
    Checks whether the gesture performed is in the config buffer i.e. recently performed.
    For most gestures, if it is present, then that makes it an invalid gesture.
    This is to prevent multiple gesture detections in a short span of time.
    '''
    if config in ['bad', 'seven', 'spiderman', 'Shake']: # these gestures are always valid, even if repeated
        return True
    if config in config_buffer: # repeated gesture
        return False
    return True
