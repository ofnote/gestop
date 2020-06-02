import zmq
from proto import landmarkList_pb2
import pyautogui
import os

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.01
context = zmq.Context()
sock = context.socket(zmq.PULL)
sock.connect("tcp://127.0.0.1:5556")

landmarkList = landmarkList_pb2.LandmarkList()

while(True):
    data = sock.recv()
    landmarkList.ParseFromString(data);
    #for l in landmarkList.landmark:
    #    print(l.x,l.y,l.z)
    index_pointer = landmarkList.landmark[8].x,landmarkList.landmark[8].y;

    resolution = pyautogui.size().width, pyautogui.size().height
    scaled_pointer = resolution[0]*index_pointer[0], resolution[1]*index_pointer[1]
    #print(scaled_pointer)
    pyautogui.moveTo(scaled_pointer[0],scaled_pointer[1],0)
    #os.system("xdotool mousemove " + str(scaled_pointer[0]) + " "  + str(scaled_pointer[1]))
