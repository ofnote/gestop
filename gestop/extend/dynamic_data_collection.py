'''
This script receives the hand keypoints detected by the keypoint generator
and then writes them to disk to create a gesture dataset.
To be run repeatedly for each gesture.
'''

import logging
import os
import argparse
import socket
import threading
from ..proto import landmarkList_pb2
from ..config import State, package_directory
from ..util.utils import start_key_listener

def main():
    ''' Main '''

    parser = argparse.ArgumentParser(description='A program to collect dynamic hand gesture data to train a neural network.')
    parser.add_argument("--user-gesture-directory", help="The directory in which gesture data should be stored. \
                        Created if it does not exist. Each gesture is stored in a separate directory inside this one.", required=True)
    args = parser.parse_args()

    # using None because arguments are irrelevant
    S = State(None, None)

    # setup socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    HOST = '127.0.0.1'
    PORT = 5556
    sock.bind((HOST, PORT))
    sock.listen(1)

    landmark_list = landmarkList_pb2.LandmarkList()

    print("Waiting for keypoint generator..")
    # Establish connection
    conn, addr = sock.accept()

    gesture = input("Enter the name of the gesture for which you are capturing data, \
    (a simple description of the gesture you will perform) :\n")

    logging.info("Hold and release the Ctrl key to record one gesture. Hit the Esc key to stop recording.")

    if not os.path.exists(args.user_gesture_directory): os.mkdir(args.user_gesture_directory)
    if not os.path.exists(os.path.join(args.user_gesture_directory, gesture)):
        os.mkdir(os.path.join(args.user_gesture_directory, gesture))

    count = 1
    start_key_listener(S)
    keypoint_buffer = []

    while True:
        data = conn.recv(4096)

        # Start recording data
        if S.ctrl_flag:
            landmark_list.ParseFromString(data)
            landmarks = []
            for lmark in landmark_list.landmark:
                landmarks.extend([str(lmark.x), str(lmark.y), str(lmark.z)])

            keypoint_buffer.append(landmarks)

        # if there is data recorded
        if len(keypoint_buffer) != 0 and not S.ctrl_flag:
            fname = os.path.join(args.user_gesture_directory, gesture, '%s%s.txt' %(gesture, count))
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

                logging.info("Gesture has been successfully recorded in {0}. Sequence len: {1}".format(
                    fname, str(len(keypoint_buffer))))
                count += 1
            else:
                logging.info("Data was not recorded properly, not written to file.")
            # Empty the buffer
            keypoint_buffer = []

        if threading.active_count() == 1:
            break

    conn.close()
    sock.close()

if __name__ == '__main__':
    main()
