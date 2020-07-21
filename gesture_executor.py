'''
Functions which execute an action given a gesture config
'''
import logging
import subprocess
from pynput.keyboard import Controller

def config_action(config, S, C):
    '''
    Given a configuration, decides what action to perform.
    '''
    if not S.ctrl_flag:
        valid = valid_config(config, S.static_config_buffer)
        S.static_config_buffer.append(config)
        S.static_config_buffer.pop(0)
        if not valid:
            config = 'bad'

    # arguments = {'keyboard':Controller(),
    #              'none':None,
    #              'state':S}

    try:
        action = C.gesture_action_mapping[config]
    except KeyError:
        logging.info("The gesture "+ config +" does not have any \
        action defined. Check the configuration file.")
        return S
    if action[0] == 'sh':  #shell
        cmd = action[1].split()
        subprocess.run(cmd, check=True)
    else: #python
        try:
            method = getattr(C.user_config, action[1])
            S = method(S)
        except AttributeError:
            logging.info("The method "+action[1]+" does not exist in user_config.py")
    return S

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
