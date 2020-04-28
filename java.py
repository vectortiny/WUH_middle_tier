import os
import datetime
import sys
import time
import subprocess
import argparse
import socket


# silmulates the WUH-JAVA-Backend sending messages to model server

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""
    This script captures photos of the hands and makes image classification.
    """)
    parser.add_argument("step", help="Hand washing step")

    args = parser.parse_args()

    STEP = args.step

msgFromClient       = str(STEP)
bytesToSend         = msgFromClient.encode()
serverAddressPort   = ("127.0.0.1", 9150)
bufferSize          = 1024

# Create a UDP socket at client side
UDPClientSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)

# Send to server using created UDP socket
UDPClientSocket.sendto(bytesToSend, serverAddressPort)

msgFromServer = UDPClientSocket.recvfrom(bufferSize)

msg = msgFromServer[0].decode()

print(msg)
