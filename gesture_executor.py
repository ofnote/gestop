'''
Functions which execute an action given a gesture config
'''
import logging
import subprocess

def pose_action(pose, S, C):
    '''
    Given a pose, decides what action to perform.
    '''
    if not S.ctrl_flag:
        valid = valid_pose(pose, S.static_pose_buffer)
        S.static_pose_buffer.append(pose)
        S.static_pose_buffer.pop(0)
        if not valid:
            pose = 'bad'

    try:
        action = C.gesture_action_mapping[pose]
    except KeyError:
        logging.info("The gesture "+ pose +" does not have any \
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

def valid_pose(pose, pose_buffer):
    '''
    Checks whether the gesture performed is in the pose buffer i.e. recently performed.
    For most gestures, if it is present, then that makes it an invalid gesture.
    This is to prevent multiple gesture detections in a short span of time.
    '''
    if pose in ['bad', 'seven', 'spiderman']: # these gestures are always valid, even if repeated
        return True
    if pose in pose_buffer: # repeated gesture
        return False
    return True
