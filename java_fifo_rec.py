import os
import sys

path = "./pipe/modelup"
fifo = open(path, "r")
for line in fifo:
    print("Received: " + line)
fifo.close()
