'''
Functions which execute an action given a gesture config
'''
import subprocess
import pyautogui
from pynput.keyboard import Controller

import user_config

def config_action(config, S, C):
    '''
    Given a configuration, decides what action to perform.
    '''
    if S.modes[0] == 'mouse':
        return config_static_action(config, S)
    else:
        return config_dynamic_action(config, S, C)

def config_static_action(config, S):
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
    valid = valid_config(config, S.static_config_buffer) #check if valid gesture
    S.static_config_buffer[S.iter%5] = config  #adding the new config to the buffer

    if config == 'bad' or not valid:
        pyautogui.mouseUp()
        S.mouse_flags['mousedown'] = False
        S.mouse_flags['scroll'] = False
    elif config == 'hitchhike':
        # Rotating list
        S.modes = S.modes[-1:] + S.modes[:-1]
    elif config == 'fist':
        S.mouse_track = not(S.mouse_track) # toggle
    elif config == 'seven':
        pyautogui.mouseDown()
        S.mouse_flags['mousedown'] = True
        S.mouse_flags['scroll'] = False
    elif config in ['four', 'eight', 'spiderman']:
        pyautogui.mouseUp()
        S.mouse_flags['mousedown'] = False
        if config == 'four':
            pyautogui.rightClick()
        elif config == 'eight':
            pyautogui.doubleClick()
        else: #spiderman
            S.mouse_flags['scroll'] = True

    return S


def config_dynamic_action(config, S, C):
    '''
    Given a gesture, executes the corresponding action.
    '''
    arguments = {'keyboard':Controller(),
                 'none':None,
                 'state':S}

    try:
        action = C.gesture_action_mapping[config]
    except KeyError:
        print("The gesture "+ config +" does not have any \
        action defined. Check the configuration file.")
        return arguments['state']
    if action[0] == 'sh':  #shell
        cmd = action[1].split()
        subprocess.run(cmd, check=True)
    else: #python
        try:
            method = getattr(user_config, action[1])
            arg = arguments[action[2]]
            arguments[action[2]] = method(arg)
        except AttributeError:
            print("The method "+action[1]+" does not exist in user_config.py")
        except KeyError:
            print("The argument "+action[2]+" is not defined. Available arguments are: "
                  + arguments.keys() + "\n. For arbitary values to be passed, wrap them in [].")
        except TypeError:
            method(action[2])
    return arguments['state']

def valid_config(config, config_buffer):
    '''
    Checks whether the gesture performed is in the config buffer i.e. recently performed.
    For most gestures, if it is present, then that makes it an invalid gesture.
    This is to prevent multiple gesture detections in a short span of time.
    '''
    if config in ['bad', 'seven', 'spiderman']: # these gestures are always valid, even if repeated
        return True
    if config in config_buffer: # repeated gesture
        return False
    return True
