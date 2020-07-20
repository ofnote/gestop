'''
This script receives the hand keypoints detected by mediapipe through
zmq and then writes them to a csv file to create a gesture dataset.
To be run repeatedly for each gesture
'''

import logging
import time
import zmq
from proto import landmarkList_pb2

def dataset_headers():
    '''
    Returns the headers of the dataset
    Header Format -> L0X,L0Y,L0Z,L1X......L20X,L20Y,L20Z,HAND
    '''
    data_str = ''
    # 21 - number of landmarks
    for i in range(21):
        data_str += 'L'+str(i)+'X,'+'L'+str(i)+'Y,'+'L'+str(i)+'Z,'
    data_str += 'HAND,GESTURE\n'
    return data_str


def add_row(landmarks, handedness, gesture, actual_hand, ROWS_ADDED):
    '''
    Formats the input data in CSV style, and returns a comma separated string i.e. a row
    '''
    row_str = ''
    if actual_hand != handedness or (landmarks[0]['x'] == 0 and
                                     landmarks[0]['y'] == 0): #wrong capture
        ROWS_ADDED -= 1 #to counter the increment in the main loop
    else:
        for i in range(len(landmarks)):
            row_str += str(landmarks[i]['x'])+','+str(landmarks[i]['y'])+ \
            ','+str(landmarks[i]['z'])+','

        row_str += str(handedness)+','+gesture+'\n'
    return row_str, ROWS_ADDED


def main():
    ''' Main '''
   
    # Setting up connection
    context = zmq.Context()
    sock = context.socket(zmq.PULL)
    sock.connect("tcp://127.0.0.1:5556")

    landmarkList = landmarkList_pb2.LandmarkList()


    actual_hand = int(input("Enter the hand for which you are collecting \
    gesture data:\n0) left \t1) right\n"))

    gesture = input("Enter the name of the gesture for which you are capturing data, \
    (a simple one word description of the orientation of your hand) :\n")

    logging.info("The script will begin executing in 5 seconds. Make sure your hand is in position.")
    time.sleep(5)

    # no. of samples added and number of samples being collected
    ROWS_ADDED = 0
    NSAMPLES = 1000

    f = open("data/static_gestures_data.csv", 'a+')
    #set pointer at beginning of file
    f.seek(0)

    # The string which is written to the dataset
    DATASET_STR = ''

    # If the file is empty, add the headers at the top of the file
    if f.read() == '':
        DATASET_STR += dataset_headers()

    while ROWS_ADDED < NSAMPLES:
        data = sock.recv()

        landmarkList.ParseFromString(data)
        landmarks = []
        for lmark in landmarkList.landmark:
            landmarks.append({'x': lmark.x, 'y': lmark.y, 'z': lmark.z})

        # Handedness - true if right hand, false if left
        handedness = landmarkList.handedness

        # Add a row to the dataset
        row_str, ROWS_ADDED = add_row(landmarks, handedness, gesture, actual_hand, ROWS_ADDED)
        DATASET_STR += row_str

        ROWS_ADDED += 1
        #simple loading bar
        print(str(ROWS_ADDED)+'/'+str(NSAMPLES)+'\t|'+('-'*int((50*ROWS_ADDED)/NSAMPLES))+'>', end='\r')

    # Writing data to file at once instead of in for loop for performance reasons.
    f.write(DATASET_STR)
    f.close()
    logging.info("1000 rows of data has been successfully collected.")


if __name__ =='__main__':
    main()
