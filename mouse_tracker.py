'''
Functions which implement code to track the mouse,
given a set of landmarks.
'''
import pyautogui


def get_avg_pointer_loc(pointer_buffer):
    '''Gets average of previous 5 pointer locations'''
    x = [i[0] for i in pointer_buffer]
    y = [i[1] for i in pointer_buffer]
    return sum(x)/len(pointer_buffer), sum(y)/len(pointer_buffer)


def calc_pointer(landmarks, C):
    ''' Uses the landmarks to calculate the location of the cursor on the screen. '''

    # The tip of the index pointer is the eighth landmark in the list
    index_pointer = landmarks[8]['x'], landmarks[8]['y'], landmarks[8]['z']

    # Screen resolution
    resolution = pyautogui.size().width, pyautogui.size().height
    scaled_pointer = resolution[0]*index_pointer[0], resolution[1]*index_pointer[1]

    C['pointer_buffer'][C['iter']%5] = scaled_pointer
    actual_pointer = get_avg_pointer_loc(C['pointer_buffer'])

    return actual_pointer, C


def mouse_track(current_pointer, C):
    '''
    Performs mouse actions depending on the C['flags'] that have been set.
    C['prev_pointer'] is only modified if the mouse is up and we are not scrolling.
    '''

    threshold = 100

    # If mouse is down and movement below threshold, do not move the mouse
    if C['flags']['mousedown'] and (abs(current_pointer[0] - C['prev_pointer'][0]) +
                               abs(current_pointer[1] - C['prev_pointer'][1]) < threshold):
        pass
    elif C['flags']['scroll']:
        amt_to_scroll = (current_pointer[1] - C['prev_pointer'][1])/10
        pyautogui.scroll(amt_to_scroll)
    else:
        pyautogui.moveTo(current_pointer[0], current_pointer[1], 0)
        C['prev_pointer'] = current_pointer
    return C
