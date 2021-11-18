# API Reference

## gestop.receiver

The server / gesture receiver. Receives gestures from the keypoint generator and processes them to perform the corresponding action. 

```sh
usage: python -m gestop.receiver [-h] [--no-mouse-track] [--config-path-json CONFIG_PATH_JSON] [--config-path-py CONFIG_PATH_PY] [--no-action]

optional arguments:
  -h, --help            show this help message and exit
  --no-mouse-track      Do not track mouse on startup
  --config-path-json CONFIG_PATH_JSON
                        Path to custom json configuration file
  --config-path-py CONFIG_PATH_PY
                        Path to custom python configuration file
  --no-action           Disbaled execution of actions. Useful for debugging.
```

## gestop.keypoint_gen.hand_tracker

Keypoint generator using mediapipe. Connects to the `receiver`, captures and sends keypoints for it to use.

```sh
python -m gestop.keypoint_gen.hand_tracker
```

## gestop.extend.static_data_collection

Script to collect data for static gestures. Used in tandem with a keypoint generator such as `gestop.keypoint_gen.hand_tracker`. Writes gesture data to a csv file for future use.

```sh
usage: python -m gestop.extend.static_data_collection [-h] [--nsamples NSAMPLES] --static-gesture-filepath STATIC_GESTURE_FILEPATH

arguments:
  --static-gesture-filepath STATIC_GESTURE_FILEPATH
                        Path to the file containing existing static gestures or path to new file to add gesture data.
optional arguments:
  -h, --help            show this help message and exit
  --nsamples NSAMPLES   The number of samples of data to collect in one run.
```

## gestop.extend.static_train_model

Script to train a neural network to detect static gestures. 

```sh
usage: python -m gestop.extend.static_train_model [-h] --static-gesture-filepath STATIC_GESTURE_FILEPATH

arguments:
  --static-gesture-filepath STATIC_GESTURE_FILEPATH
                        Path to the file containing static gestures.
optional arguments:
  -h, --help            show this help message and exit
```

## gestop.extend.dynamic_data_collection

Script to collect data for dynamic gestures. Used in tandem with a keypoint generator such as `gestop.keypoint_gen.hand_tracker`. Writes gesture data to a directory for future use.

```sh
usage: python -m gestop.extend.dynamic_data_collection [-h] --user-gesture-directory USER_GESTURE_DIRECTORY

arguments:
  --user-gesture-directory USER_GESTURE_DIRECTORY
                        The directory in which gesture data should be stored. Created if it does not exist. Each gesture is stored in a separate directory inside this one.

optional arguments:
  -h, --help            show this help message and exit
```

## gestop.extend.dynamic_train_model

Script to train a neural network to detect static gestures. 

```sh
usage: python -m gestop.extend.dynamic_train_model [-h] [--exp-name EXP_NAME] --shrec-directory SHREC_DIRECTORY [--user-directory USER_DIRECTORY] [--use-pretrained]

arguments:
  --shrec-directory SHREC_DIRECTORY
                        The directory of SHREC.

optional arguments:
  -h, --help            show this help message and exit
  --exp-name EXP_NAME   The name with which to log the run.
  --user-directory USER_DIRECTORY
                        The directory in which user collected gesture data is stored.
  --use-pretrained      Use pretrained model.
```

## gestop.util.find_keycode

A utility script to find the keycodes when a particular key is pressed. Useful when creating a custom config which makes use of special keys (e.g. media\_volume\_up)

```sh
python -m gestop.util.find_keycode
```
