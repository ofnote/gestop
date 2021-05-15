'''
The definition of the functions which will be executed when the
corresponding gesture is detected.
'''

from pynput import mouse, keyboard
from pynput.keyboard import Key, KeyCode
from pynput.mouse import Button

class UserConfig:
    ''' Class for user configuration.'''
    def __init__(self):
        self.mouse = mouse.Controller()
        self.keyboard = keyboard.Controller()

    def no_func(self, S):
        ''' Dummy function. '''
        return S

    def take_screenshot(self, S):
        ''' Takes a screenshot. '''
        self.keyboard.press(Key.print_screen)
        self.keyboard.release(Key.print_screen)
        return S

    def inc_volume(self, S):
        ''' Increase system volume. '''
        self.keyboard.press(Key.media_volume_up)
        self.keyboard.release(Key.media_volume_up)
        return S

    def dec_volume(self, S):
        ''' Decrease system volume. '''
        self.keyboard.press(Key.media_volume_down)
        self.keyboard.release(Key.media_volume_down)
        return S

    def inc_brightness(self, S):
        ''' Increase system volume. '''
        self.keyboard.press(KeyCode.from_vk(269025026))
        self.keyboard.release(KeyCode.from_vk(269025026))
        return S

    def dec_brightness(self, S):
        ''' Decrease system volume. '''
        self.keyboard.press(KeyCode.from_vk(269025027))
        self.keyboard.release(KeyCode.from_vk(269025027))
        return S

    def circle_detected(self, S):
        ''' Trial function. '''
        print("Circle has been detected.")
        return S

    def reset_mouse(self, S):
        ''' Resets the mouse flags. '''
        self.mouse.release(Button.left)
        S.mouse_flags['mousedown'] = False
        S.mouse_flags['scroll'] = False
        return S

    def disable_mouse_track(self, S):
        ''' Disables tracking of mouse. '''
        S.mouse_track = not S.mouse_track
        return S

    def left_mouse_down(self, S):
        ''' The left mouse button is clicked and held. '''
        self.mouse.press(Button.left)
        S.mouse_flags['mousedown'] = True
        S.mouse_flags['scroll'] = False
        return S

    def double_click(self, S):
        ''' Left mouse button double click. '''
        self.mouse.release(Button.left)
        S.mouse_flags['mousedown'] = False
        self.mouse.click(Button.left, 2)
        return S

    def right_mouse_click(self, S):
        ''' Right mouse button click. '''
        S.mouse_flags['mousedown'] = False
        self.mouse.click(Button.right)
        return S

    def left_mouse_click(self, S):
        ''' Right mouse button click. '''
        S.mouse_flags['mousedown'] = False
        self.mouse.click(Button.left)
        return S


    def scroll(self, S):
        ''' Locks the mouse and allows scrolling. '''
        self.mouse.release(Button.left)
        S.mouse_flags['mousedown'] = False
        S.mouse_flags['scroll'] = True
        return S
