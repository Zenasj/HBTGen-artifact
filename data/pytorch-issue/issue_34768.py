#!/usr/bin/env python3
import sys
import socket
import os
import array
import shutil
import socket


if len(sys.argv) != 4:
    print("Usage: ", sys.argv[0], " tmp_dirname iteration (send|recv)")
    sys.exit(1)

if __name__ == '__main__':
    dirname = sys.argv[1]
    sock_path = dirname + "/sock"
    iterations = int(sys.argv[2])
    def dummy_path(i):
        return dirname + "/" + str(i) + ".dummy"


    if sys.argv[3] == 'send':
        while not os.path.exists(sock_path):
            pass
        client = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        client.connect(sock_path)
        for i in range(iterations):
            fd = os.open(dummy_path(i), os.O_WRONLY | os.O_CREAT)
            ancdata = array.array('i', [fd])
            msg = bytes([i % 256])
            print("Sending fd ", fd, " (iteration #", i, ")")
            client.sendmsg([msg], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, ancdata)])


    else:
        assert sys.argv[3] == 'recv'

        if os.path.exists(dirname):
            raise Exception("Directory exists")

        os.mkdir(dirname)

        print("Opening socket...")
        server = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        server.bind(sock_path)

        print("Listening...")
        for i in range(iterations):
            a = array.array('i')
            msg, ancdata, flags, addr = server.recvmsg(1, socket.CMSG_SPACE(a.itemsize))
            assert(len(ancdata) == 1)
            cmsg_level, cmsg_type, cmsg_data = ancdata[0]
            a.frombytes(cmsg_data)
            print("Received fd ", a[0], " (iteration #", i, ")")

        shutil.rmtree(dirname)