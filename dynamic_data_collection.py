'''
This script receives the hand keypoints detected by mediapipe through
zmq and then writes them to disk to create a gesture dataset.
To be run repeatedly for each gesture
'''

import os
import threading
import zmq
from proto import landmarkList_pb2
from config import State
from gesture_receiver import start_key_listener

def main():
    ''' Main '''

    # using None because arguments are irrelevant
    S = State(None, None)

    # Setting up connection
    context = zmq.Context()
    sock = context.socket(zmq.PULL)
    sock.connect("tcp://127.0.0.1:5556")

    landmarkList = landmarkList_pb2.LandmarkList()

    gesture = input("Enter the name of the gesture for which you are capturing data, \
    (a simple description of the gesture you will perform) :\n")

    print("Hold and release the Ctrl key to record one gesture. Hit the Esc key to stop recording.")

    path = "data/dynamic_gestures/" + gesture
    if not os.path.exists(path):
        os.mkdir(path)

    count = 1
    start_key_listener(S)
    keypoint_buffer = []

    while True:
        data = sock.recv()

        if S.ctrl_flag:
            landmarkList.ParseFromString(data)
            landmarks = []
            for lmark in landmarkList.landmark:
                landmarks.extend([str(lmark.x), str(lmark.y), str(lmark.z)])

            keypoint_buffer.append(landmarks)

        if len(keypoint_buffer) != 0 and not S.ctrl_flag:
            fname = path + "/" + gesture + str(count) + ".txt"
            lmark_str = ''
            for i in keypoint_buffer:
                # verifying data quality
                if '0.0' in i:
                    lmark_str = ''
                    break
                lmark_str += ' '.join(i) + '\n'

            if lmark_str != '':
                with open(fname, 'w') as f:
                    f.write(lmark_str)

                print("Gesture has been successfully recorded in " + fname + \
                      '. Sequence len:' + str(len(keypoint_buffer)))
                count += 1
            else:
                print("Data was not recorded properly, not written to file.")
            keypoint_buffer = []

        if threading.active_count() == 1:
            break

if __name__ == '__main__':
    main()
