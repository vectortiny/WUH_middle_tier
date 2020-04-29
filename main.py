import socket
import os
import datetime
import sys
import time
import subprocess
import argparse

#from tflite_runtime.interpreter import Interpreter
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# read the absolute path
script_dir = os.getcwd()

def load_labels(path):
  with open(path, 'r') as f:
    return {i: line.strip() for i, line in enumerate(f.readlines())}

def set_input_tensor(interpreter, image):
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image

def classify_image(interpreter, image, top_k=1):
    """Returns a sorted array of classification results."""
    set_input_tensor(interpreter, image)
    interpreter.invoke()
    output_details = interpreter.get_output_details()[0]
    output = np.squeeze(interpreter.get_tensor(output_details['index']))

    # If the model is quantized (uint8 data), then dequantize the results
    if output_details['dtype'] == np.uint8:
        scale, zero_point = output_details['quantization']
        output = scale * (output - zero_point)

    ordered = np.argpartition(-output, top_k)
    return [(i, output[i]) for i in ordered[:top_k]]

# main

labels = load_labels("./model/converted_tflite_quantized/labels.txt")
#interpreter = Interpreter("./model/converted_tflite_quantized/model.tflite")
interpreter = tf.lite.Interpreter("./model/converted_tflite_quantized/model.tflite")
interpreter.allocate_tensors()
_, height, width, _ = interpreter.get_input_details()[0]['shape']

localIP     = "127.0.0.1"
localPort   = 9150
bufferSize  = 1024

# Create a datagram socket
UDPServerSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)

# Bind to address and ip
UDPServerSocket.bind((localIP, localPort))

print("Model server up and listening")

# Listen for incoming datagrams
while(True):
    message, address = UDPServerSocket.recvfrom(bufferSize)
    STEP = message.decode()
    clientIP  = "Client IP Address:{}".format(address)

    # print(STEP)
    # print(clientIP)

    ##

    results=[]
    for x in range(0, 5):
        # call the .sh to capture the image
        img_path = './webcam/hand_' + str(STEP) + '_' + str(x) + '.jpg'

        os.system(
            script_dir +
            '/webcam.sh ' +
            script_dir + ' ' +
            str(STEP) + ' ' + str(x)
        )
        image = Image.open(img_path).convert('RGB').resize((width, height),Image.ANTIALIAS)
        results += classify_image(interpreter, image)
        time.sleep(0.1)

    cnt_class = 0
    for ind, pred in results:
        if int(ind) == int(STEP):
            cnt_class += 1

    #print(results)
    if cnt_class >= 3:
        msgFromServer = STEP
    else:
        msgFromServer = '-1'


    # Sending a reply to client

    bytesToSend         = str.encode(msgFromServer)

    UDPServerSocket.sendto(bytesToSend, address)
