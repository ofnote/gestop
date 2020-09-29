'''
Gesture Executor - Executes the action corresponding to a detected gesture
'''
import logging
import subprocess

def pose_action(pose, S, C):
    '''
    Given a pose, executes the corresponding action
    '''
    try:
        action = C.gesture_action_mapping[pose]
    except KeyError:
        logging.info("The gesture "+ pose +" does not have any \
        action defined. Check the configuration file.")
        return S

    if not S.ctrl_flag:
        valid = valid_action(action[1], S.static_action_buffer)
        S.static_action_buffer.append(action[1])
        S.static_action_buffer.pop(0)
        if not valid:
            action = ['py', 'reset_mouse']

    if action[0] == 'sh':  #shell
        cmd = action[1].split()
        subprocess.run(cmd, check=True)
    else: #python
        try:
            method = getattr(C.user_config, action[1])
            S = method(S)
        except AttributeError:
            logging.info("The method "+action[1]+" does not exist in user_config.py")
    S.action = action[1]
    return S

def valid_action(action, action_buffer):
    '''
    Checks whether the action is in the action buffer i.e. recently executed.
    For most actions, if it is present, then that makes it invalid.
    This is to prevent multiple actions in a short span of time.
    '''
    if action in ['reset_mouse', 'scroll', 'left_mouse_down']:
        return True
    if action in action_buffer: # repeated action
        return False
    return True
