import socket
import struct
from proto import landmarkList_pb2
import socketserver
import time
import threading


def get_landmarks(data, landmarkList):
    landmarkList.ParseFromString(data)
    landmarks = []
    for lmark in landmarkList.landmark:
        landmark = ({'x': lmark.x, 'y': lmark.y, 'z': lmark.z})
        #print(landmark)
        landmarks.append(landmark)
    #print(landmarks)

    return landmarks, landmarkList.handedness

 


from gesture_receiver import all_init, process_data

class MyTCPHandler(socketserver.BaseRequestHandler):
    """
    The request handler class for our server.

    It is instantiated once per connection to the server, and must
    override the handle() method to implement communication to the
    client.
    """

    def setup(self):
        self.landmark_list = landmarkList_pb2.LandmarkList()
        self.C, self.S, landmark_list = all_init()


    def handle_data(self, data):
        landmarks, handedness = get_landmarks(data, self.landmark_list)
        #for l in landmarks: print(l)
        #print(handedness)
        #time.sleep(5)
        
        process_data(data, self.landmark_list, self.C, self.S)
        

    def handle(self):
        # self.request is the TCP socket connected to the client
        print("{} wrote:".format(self.client_address[0]))
        count_empty = 0
        while True:
            msg_len = self.request.recv(4)
            msg_len = struct.unpack("I", msg_len)
            #print(msg_len)
            data = self.request.recv(msg_len[0])
            #print(f'--[{len(self.data)}]')

            # detect empty data (alias for client disconnected)
            if data == b'': 
                count_empty += 1
            if count_empty > 100 : break

            self.handle_data(data)

            # The key listener thread has shut down, leaving only GestureThread and MainThread
            if threading.active_count() == 1:
                break

            # just send back the same data, but upper-cased
            #self.request.sendall(self.data.upper())



def run_socket_server():

    HOST, PORT = "0.0.0.0", 8089

    # Create the server, binding to localhost on port 9999
    with socketserver.TCPServer((HOST, PORT), MyTCPHandler) as server:
        print(f'Server now listening {PORT}')
        server.serve_forever()

if __name__ == "__main__":
    run_socket_server()
