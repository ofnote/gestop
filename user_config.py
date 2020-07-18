'''
The definition of the functions which will be executed when the
corresponding gesture is detected.
'''

from pynput.keyboard import Key, KeyCode


def no_func(none):
    ''' Dummy function. '''
    return none

def take_screenshot(keyboard):
    ''' Takes a screenshot. '''
    keyboard.press(Key.print_screen)
    keyboard.release(Key.print_screen)
    return keyboard

def mode_switch(S):
    ''' Switches back to mouse mode. '''
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
