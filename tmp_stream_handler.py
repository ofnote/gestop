import socket
import struct
from proto import landmarkList_pb2
import socketserver



def get_landmarks(data, landmarkList):
    landmarkList.ParseFromString(data)
    landmarks = []
    for lmark in landmarkList.landmark:
        landmark = ({'x': lmark.x, 'y': lmark.y, 'z': lmark.z})
        #print(landmark)
        landmarks.append(landmark)
    #print(landmarks)

    return landmarks, landmarkList.handedness

 

class MyTCPHandler(socketserver.BaseRequestHandler):
    """
    The request handler class for our server.

    It is instantiated once per connection to the server, and must
    override the handle() method to implement communication to the
    client.
    """

    def setup(self):
        self.LandmarkList = landmarkList_pb2.LandmarkList()

    def handle(self):
        import time
        # self.request is the TCP socket connected to the client
        print("{} wrote:".format(self.client_address[0]))
        count_empty = 0
        while True:
            msg_len = self.request.recv(4)
            msg_len = struct.unpack("I", msg_len)
            #print(msg_len)
            self.data = self.request.recv(msg_len[0])
            #print(f'--[{len(self.data)}]')

            # detect empty data (alias for client disconnected)
            if self.data == b'': 
                count_empty += 1
            if count_empty > 100 : break

            landmarks, handedness = get_landmarks(self.data, self.LandmarkList)
            for l in landmarks: print(l)
            print(handedness)
            time.sleep(5)

            # just send back the same data, but upper-cased
            #self.request.sendall(self.data.upper())


def initialize_gesture_recognizer():
    pass

def run_socket_server():
    initialize_gesture_recognizer()

    HOST, PORT = "0.0.0.0", 8089

    # Create the server, binding to localhost on port 9999
    with socketserver.TCPServer((HOST, PORT), MyTCPHandler) as server:
        print(f'Server now listening {PORT}')
        server.serve_forever()

if __name__ == "__main__":
    run_socket_server()
