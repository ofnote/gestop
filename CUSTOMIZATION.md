# Customization

**gestop** is highly customizable and can be easily extended in various ways. The existing gesture-action pairs can be remapped, new actions can be defined (either a python function or a shell script, opening up a world of possiiblity to interact with your computer), and finally, if you so desire, you can capture data to create your own gestures and retrain the network to utilize your own custom gestures. The ways to accomplish the above are briefly described in this section. 

The default gesture-action mappings are stored in `data/action_config.json`. The format of the config file is:

`{'gesture_name':['type','func_name']}`

Where, gesture_name is the name of the gesture that is detected, type is either `sh`(shell) or `py`(python). If the type is `py`, then `func_name` is the name of a python function and if the type is `sh`, then `func_name` is either a shell command or a shell script (`./path/to/shell_script.sh`). Refer `data/action_config.json` and `executor.py` for more details.

It is encouraged to make all custom configuration in a new file rather than replace the original. So, before your modifications, copy `data/action_config.json` and create a new file. After your modifications are done in the new file, you can run the application with your custom config using `python -m receiver.py --config-path my_custom_config.json`

### Remap gestures

To remap functionality, all you need to do is swap the values (i.e. ``['type','func_name']`) for the gestures you wish to remap. As an example if you wish to take a screenshot with `Swipe +`, instead of `Grab`, the configuration would change from:

``` json
    "Grab" : ["py", "take_screenshot"],
    "Swipe +" : ["py", "no_func"],
```

To,

``` json
    "Grab" : ["py", "no_func"],
    "Swipe +" : ["py", "take_screenshot"],
```

### Adding new actions

Adding new actions is a similar process to remapping gestures, except for the additional step of defining your python function/shell command. As a simple example, if you wish to type your username on performing `Pinch`, the first step would be to write the python function in `user_config.py`. The function would be something similar to the following:

``` python
def print_username(self, S):
    ''' Prints username '''
    self.keyboard.type("sriramsk1999")
    return S
```

Where `S` represents the *State* and is passed to all the functions in `user_config.py`. Refer `user_config.py` and `config.py` to see more examples of how to add new actions. 

Finally, replace the existing `Pinch` mapping with your own in your configuration file.

``` json

    "Pinch" : ["py", "print_username"],
```

### Adding new gestures

To extend this application and create new gestures, there are a few prerequisites. Firstly, download the data from from the dataset link given. This is to ensure that your model has all existing data along with the new data to train on.

You can either record a new static gesture or a dynamic gesture, with the `static_data_collection.py` and `dynamic_data_collection.py` scripts respectively.

To collect data for a new static gesture, run the program, enter the name of the gesture and the hand with which you will be performing the gesture. Run the Python or C++ keypoint generators and hold the gesture while data is collected. A 1000 samples are collected which should take a minute or two. Hold your hand in the same pose in good lighting to ensure the model gets clean data.

To collect data for a new dynamic gesture, the process is mostly similar. Run the `dynamic_data_collection.py` program, enter the name of the gesture and run the keypoint generator. Data is collected only when the Ctrl key is held down, so to collect a single sample, hold the Ctrl key, perform the gesture and then release. Repeat this process a few dozen times to collect enough data.

The next step is to retrain the network, using the `static_train_model.py` or the `dynamic_train_model.py` script depending on the new gesture. Finally, add the new gesture-action mapping to the configuration file. And that's it! Your new gesture is now part of gestop. 

Refer to [API Reference](API_REFERENCE.md) for more details.
