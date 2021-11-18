'''
The client application. Generates keypoints using mediapipe's Python API and
transmits them using sockets.
'''
import socket
import cv2
import mediapipe as mp
from ..proto import landmarkList_pb2

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

HOST = '127.0.0.1'
PORT = 5556
try:
    sock.connect((HOST, PORT))
except Exception as e:
    print("Server Connection Failed ", e)
    exit(0)

# For webcam input:
hands = mp_hands.Hands(
    min_detection_confidence=0.4, min_tracking_confidence=0.4)
cap = cv2.VideoCapture(0)
while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = hands.process(image)

    landmarkList = landmarkList_pb2.LandmarkList()

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        handedness = results.multi_handedness[0].classification[0].index

        landmarkList.handedness = handedness
        for lmark in hand_landmarks.landmark:
            l = landmarkList.landmark.add()
            l.x = lmark.x
            l.y = lmark.y
            l.z = lmark.z*256

        output = landmarkList.SerializeToString()
        try: sock.send(output)
        except BrokenPipeError: break

        mp_drawing.draw_landmarks(
              image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

hands.close()
cap.release()
sock.close()
