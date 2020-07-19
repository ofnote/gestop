'''
The definition of the functions which will be executed when the
corresponding gesture is detected.
'''

from pynput.keyboard import Key, KeyCode
import pyautogui


def no_func(none):
    ''' Dummy function. '''
    return none

def take_screenshot(keyboard):
    ''' Takes a screenshot. '''
    keyboard.press(Key.print_screen)
    keyboard.release(Key.print_screen)
    return keyboard

def mode_switch(S):
    ''' Switches modes. '''
    S.modes = S.modes[-1:] + S.modes[:-1]
    return S

def inc_volume(keyboard):
    ''' Increase system volume. '''
    keyboard.press(KeyCode.from_vk(269025043))
    keyboard.release(KeyCode.from_vk(269025043))
    return keyboard

def dec_volume(keyboard):
    ''' Decrease system volume. '''
    keyboard.press(KeyCode.from_vk(269025041))
    keyboard.release(KeyCode.from_vk(269025041))
    return keyboard

def inc_brightness(keyboard):
    ''' Increase system volume. '''
    keyboard.press(KeyCode.from_vk(269025026))
    keyboard.release(KeyCode.from_vk(269025026))
    return keyboard

def dec_brightness(keyboard):
    ''' Decrease system volume. '''
    keyboard.press(KeyCode.from_vk(269025027))
    keyboard.release(KeyCode.from_vk(269025027))
    return keyboard

def circle_detected(none):
    ''' Trial function. '''
    print("Circle has been detected.")
    return none

def reset_mouse(S):
    ''' Resets the mouse flags. '''
    pyautogui.mouseUp()
    S.mouse_flags['mousedown'] = False
    S.mouse_flags['scroll'] = False
    return S

def disable_mouse_track(S):
    ''' Disables tracking of mouse. '''
    S.mouse_track = not S.mouse_track
    return S

def left_mouse_down(S):
    ''' The left mouse button is clicked and held. '''
    pyautogui.mouseDown()
    S.mouse_flags['mousedown'] = True
    S.mouse_flags['scroll'] = False
    return S

def double_click(S):
    ''' Left mouse button double click. '''
    pyautogui.mouseUp()
    S.mouse_flags['mousedown'] = False
    pyautogui.doubleClick()
    return S

def right_mouse_click(S):
    ''' Right mouse button click. '''
    pyautogui.mouseUp()
    S.mouse_flags['mousedown'] = False
    pyautogui.rightClick()
    return S

def scroll(S):
    ''' Locks the mouse and allows scrolling. '''
    pyautogui.mouseUp()
    S.mouse_flags['mousedown'] = False
    S.mouse_flags['scroll'] = True
    return S
