import zmq
from proto import landmarkList_pb2
import pyautogui
import math

#calculate the angle that (x1,y1) and (x2,y2) make at (x0,y0) 
def angleBetweenLines(x0,y0,x1,y1,x2,y2,handedness):
    angle1 = math.degrees(math.atan2(y0-y1,x0-x1))
    angle2 = math.degrees(math.atan2(y0-y2,x0-x2))

    if (handedness): return angle1-angle2
    else: return angle2-angle1
    
# calculates the various angles from the joints of the hand, see `Useful Information` in README.md for more details
# PIP (Proximal InterPhalangeal) angles - lower joint angles (5)
# DIP (Dostal InterPhalangeal) angles - upper joint angles (5)
# MCP (MetaCarpoPhalangeal) angles - angles between fingers (4)
# Palm angle - rotation of the hand with respect to the vertical axis (1)
def calculateAngles(landmarks, handedness):
    angles = {}
    angles['dip'] = []
    angles['pip'] = []
    angles['mcp'] = []
    for i in range(5):
        angles['dip'].append(angleBetweenLines(landmarks[3+(4*i)]['x'],landmarks[3+(4*i)]['y'],
                                               landmarks[3+(4*i)+1]['x'],landmarks[3+(4*i)+1]['y'],
                                               landmarks[3+(4*i)-1]['x'],landmarks[3+(4*i)-1]['y'],
                                               handedness)) #L3,L7,L11,L15,L19
        angles['pip'].append(angleBetweenLines(landmarks[2+(4*i)]['x'],landmarks[2+(4*i)]['y'],
                                               landmarks[2+(4*i)+1]['x'],landmarks[2+(4*i)+1]['y'],
                                               landmarks[2+(4*i)-1]['x'],landmarks[2+(4*i)-1]['y'],
                                               handedness)) #L2,L6,L10,L14,L18


    for i in range(4):
        angles['mcp'].append(angleBetweenLines(landmarks[1+(4*i)]['x'],landmarks[1+(4*i)]['y'],
                                               landmarks[1+(4*i)+3]['x'],landmarks[1+(4*i)+3]['y'],
                                               landmarks[1+(4*i)+7]['x'],landmarks[1+(4*i)+7]['y'],
                                               handedness)) #L1,L5,L9,L13

    angles['palm'] = angleBetweenLines(landmarks[0]['x'],landmarks[0]['y'],
                                       landmarks[9]['x'],landmarks[9]['y'],
                                       0,landmarks[0]['y'], handedness) #L2,L6,L10,L14,L18
    return angles

def getAvgPointerLoc(pointer_buffer):
    x = [i[0] for i in pointer_buffer]
    y = [i[1] for i in pointer_buffer]
    return sum(x)/len(pointer_buffer), sum(y)/len(pointer_buffer)

# allow mouse to move to edge of screen, and set interval between calls to 0.01
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.01

# Setting up connection
context = zmq.Context()
sock = context.socket(zmq.PULL)
sock.connect("tcp://127.0.0.1:5556")

landmarkList = landmarkList_pb2.LandmarkList()

# maintain a buffer of most recent movements to smoothen mouse movement
pointer_buffer = [(0,0), (0,0), (0,0), (0,0), (0,0)]
iter_count = 0

while True:
    data = sock.recv()
    landmarkList.ParseFromString(data)
    landmarks = []
    for l in landmarkList.landmark:
        landmarks.append({'x':l.x,'y':l.y,'z':l.z})

    #handedness - true if right hand, false if left
    handedness = landmarkList.handedness

    #The tip of the index pointer is the eighth landmark in the list
    index_pointer = landmarks[8]['x'], landmarks[8]['y'], landmarks[8]['z']

    #Screen resolution
    resolution = pyautogui.size().width, pyautogui.size().height
    scaled_pointer = resolution[0]*index_pointer[0], resolution[1]*index_pointer[1]

    pointer_buffer[iter_count%5] = scaled_pointer
    actual_pointer = getAvgPointerLoc(pointer_buffer)

    pyautogui.moveTo(actual_pointer[0], actual_pointer[1], 0)
    angles = calculateAngles(landmarks, handedness)

    fingerState = []
    if (angles['pip'][0] + angles['dip'][0] < 400): #thumbAngle
        fingerState.append('straight')
    else: fingerState.append('bent')

    for i in range(1,5):
        if (angles['pip'][i] + angles['dip'][i] > 0):
            fingerState.append('straight')
        else:
            fingerState.append('bent')
    print(fingerState)
    if(fingerState == ['straight', 'straight', 'bent', 'bent', 'bent']):
        pyautogui.mouseDown()
    else:
        pyautogui.mouseUp()

    iter_count+=1
