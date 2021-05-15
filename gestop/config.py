'''
Functions which initialize the 'configuration' -> values which don't change during runtime
and the 'state' -> values which represent the state of the application
'''
import json
from dataclasses import dataclass, field
import logging
from sys import stdout, platform
import os
import datetime
from typing import Dict, List, Tuple
import importlib.util
import torch
from pynput.mouse import Controller

from .model import StaticNet, DynamicNet
from .user_config import UserConfig

package_directory = os.path.dirname(os.path.abspath(__file__))

def get_screen_resolution():
    ''' OS independent way of getting screen resolution. Adapted from pyautogui. '''
    if platform == 'linux':
        from Xlib.display import Display

        _display = Display(os.environ['DISPLAY'])
        return (_display.screen().width_in_pixels, _display.screen().height_in_pixels)
    elif platform == 'darwin':
        try:
            import Quartz
        except:
            assert False, "You must first install pyobjc-core and pyobjc"
        return (Quartz.CGDisplayPixelsWide(Quartz.CGMainDisplayID()),
                Quartz.CGDisplayPixelsHigh(Quartz.CGMainDisplayID()))
    elif platform == 'win32':
        import ctypes
        return (ctypes.windll.user32.GetSystemMetrics(0), ctypes.windll.user32.GetSystemMetrics(1))

def setup_logger():
    ''' Set up logger '''
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(package_directory, "logs/debug{}.log".format(
                datetime.datetime.now().strftime("%m.%d:%H.%M.%S")))),
            logging.StreamHandler(stdout)
        ]
    )
    # Disabled to prevent debug output by matplotlib
    logging.getLogger('matplotlib.font_manager').disabled = True

def get_seed():
    ''' Seed value for reproducibility '''
    return 42

@dataclass
class Config:
    ''' The configuration of the application. '''
    if not torch.cuda.is_available():
        map_location = torch.device('cpu')
    else:
        map_location = None

    if not os.path.exists(os.path.join(package_directory, 'logs')):
        os.mkdir(os.path.join(package_directory, 'logs'))

    setup_logger()

    # If lite is true, then the neural networks are not loaded into the config
    # This is useful in scripts which do not use the network, or may modify the network.
    lite: bool

    # Path to action configuration file (json)
    config_path_json: str = os.path.join(package_directory, 'data/action_config.json')

    # Path to action configuration file (py)
    config_path_py: str = os.path.join(package_directory, 'user_config.py')

    # Refer make_vector() in train_model.py to verify input dimensions
    static_input_dim: int = 49
    static_output_classes: int = 0 # see __post_init__

    # Refer format_mediapipe() in dynamic_train_model.py to verify input dimensions
    dynamic_input_dim: int = 36
    shrec_output_classes: int = 14 # Number of gestures in SHREC
    dynamic_output_classes: int = 0 # see __post_init__

    # Minimum number of epochs
    min_epochs: int = 15

    static_batch_size: int = 64
    dynamic_batch_size: int = 1

    # value for pytorch-lighting trainer attribute accumulate_grad_batches
    grad_accum: int = 2

    static_gesture_mapping: dict = field(default_factory=dict)
    dynamic_gesture_mapping: dict = field(default_factory=dict)

    # Screen Resolution
    resolution: Tuple = get_screen_resolution()

    # Mapping of gestures to actions
    gesture_action_mapping: dict = field(default_factory=dict)

    static_path: str = os.path.join(package_directory, 'models/static_net.pth')
    dynamic_path: str = os.path.join(package_directory, 'models/dynamic_net.pth')

    static_net: StaticNet = field(init=False)
    dynamic_net: DynamicNet = field(init=False)

    # Mouse tracking
    mouse: Controller = field(init=False)
    # How much a single scroll action should scroll
    scroll_unit: int = 10

    # Specifying how to map webcam coordinates to the monitor coordinates.
    # Format - [x1,y1,x2,y2] where (x1,y1) specifies which coordinate to map to
    # the top left of your screen and (x2,y2) specifies which coordinate to map
    # to the bottom right of your screen.
    map_coord = [0.2, 0.2, 0.8, 0.8]

    # User configuration
    user_config: UserConfig = field(init=False)

    def __post_init__(self):
        logging.info("Package Directory: %s" %package_directory)

        # Fetching gesture mappings
        with open(os.path.join(package_directory, 'data/static_gesture_mapping.json'), 'r') as jsonfile:
            self.static_gesture_mapping = json.load(jsonfile)
            self.static_output_classes = len(self.static_gesture_mapping)
        with open(os.path.join(package_directory, 'data/dynamic_gesture_mapping.json'), 'r') as jsonfile:
            self.dynamic_gesture_mapping = json.load(jsonfile)
            self.dynamic_output_classes = len(self.dynamic_gesture_mapping)

        with open(self.config_path_json, 'r') as jsonfile:
            self.gesture_action_mapping = json.load(jsonfile)

        self.mouse = Controller()

        user_config_spec = importlib.util.spec_from_file_location("user_config", self.config_path_py)
        user_config = importlib.util.module_from_spec(user_config_spec)
        user_config_spec.loader.exec_module(user_config)
        self.user_config = user_config.UserConfig()

        # Setting up networks
        if not self.lite:
            logging.info('Loading StaticNet...')
            self.static_net = StaticNet(self.static_input_dim, self.static_output_classes,
                                          self.static_gesture_mapping)
            self.static_net.load_state_dict(torch.load(self.static_path,
                                                        map_location=self.map_location))
            self.static_net.eval()

            logging.info('Loading DynamicNet...')

            self.dynamic_net = DynamicNet(self.dynamic_input_dim, self.dynamic_output_classes,
                                      self.dynamic_gesture_mapping)
            self.dynamic_net.load_state_dict(torch.load(self.dynamic_path,
                                                      map_location=self.map_location))
            self.dynamic_net.eval()

@dataclass
class State:
    ''' The state of the application. '''

    # Flag that indicates whether mouse is tracked
    mouse_track: bool
    # Flag to indicate whether actions should be executed
    exec_action: bool

    # array of flags for mouse control
    mouse_flags: Dict = field(default_factory=dict)

    # maintain a buffer of most recent movements to smoothen mouse movement
    pointer_buffer: List = field(default_factory=list)
    prev_pointer: List = field(default_factory=list)

    # maintain a buffer of most recently executed actions
    static_action_buffer: List = field(default_factory=list)

    # maintain a buffer of keypoints for dynamic gestures
    keypoint_buffer: List = field(default_factory=list)

    # Stores the previous landmark in the stream
    prev_landmark: torch.tensor = None

    # Flag to denote whether the Ctrl key is pressed
    ctrl_flag: bool = False
    # ctrl_flag of the previous timestep. Used to detect change
    prev_flag: bool = False

    # Last executed action
    action: str = ''

    def __post_init__(self):

        self.mouse_flags = {'mousedown': False, 'scroll': False}

        self.pointer_buffer = [(0, 0) for i in range(5)]
        self.prev_pointer = [0, 0]

        self.static_action_buffer = ['', '', '', '', '']
