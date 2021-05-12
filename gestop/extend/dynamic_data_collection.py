'''
This script receives the hand keypoints detected by the keypoint generator
and then writes them to disk to create a gesture dataset.
To be run repeatedly for each gesture.
'''

import logging
import os
import socket
import threading
from ..proto import landmarkList_pb2
from ..config import State, package_directory
from ..util.utils import start_key_listener

def main():
    ''' Main '''

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

    path = os.path.join(package_directory, "data/dynamic_gestures/", gesture)
    if not os.path.exists(path):
        os.mkdir(path)

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
