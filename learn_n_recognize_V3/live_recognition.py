#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Latest file editing:
# 18-09-2016 14h50:50
#
# Copyright (c) 2016, Samuel Prevost <samuel.prevost@gmx.com>
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of Learn 'n' Recognize nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import cv2, os, sys
import numpy as np
from PIL import Image
import math
import datetime

# dataset images' names should start with :
# %%SUBJECT_NAME%%.%ID%

#Arg 1: dataset (with or without "recognizer.xml" file in it)
#Arg 2: camera to use
#Arg 3: video scale

dynamicTextScaling = True
# 3.5 default
dynamicTextScalingRate = 3.3
# 1.5 default
staticTextScalingSize = 1.5

# Load the face detection cascade from haar cascade file
# Should detect human faces in general
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

# Registrating video camera
video_capture = cv2.VideoCapture(int(sys.argv[2]))

# Setting the video scaling ratio (I'm using 2.5 for testing)
video_scale = float(sys.argv[3])

# Create a face recognizer object
# (Here we're using Local Binary Patterns Histograms (LBPH) algoritm)
recognizer = cv2.createLBPHFaceRecognizer()

# Get the subject's name based on his id
def get_name_for_id(id, path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    image_paths = [x for x in image_paths if "recognizer.xml" not in x]
    image_paths = [x for x in image_paths if "DS_Store" not in x]

    name = "[ Name Not Found ]"
    for image_path in image_paths:
        if(int(os.path.split(image_path)[1].split(".")[1]) == id):
            name = os.path.split(image_path)[1].split(".")[0]
            break;
        else:
            continue

    return name


def get_images_and_ids(path):
    # Append all the absolute image paths in a list image_paths
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    image_paths = [x for x in image_paths if "DS_Store" not in x]
    # images will contains face images
    images = []
    # ids will contains the id that is assigned to the image
    ids = []
    # names will contains the names of the person on the image
    names = []
    for image_path in image_paths:
        # Read the image and convert to grayscale
        image_pil = Image.open(image_path).convert('L')
        # Convert the image format into numpy array
        image = np.array(image_pil, 'uint8')
        # Split the filename
        splittedName = os.path.split(image_path)[1].split(".")
        # Get the name of the people on the image
        name = splittedName[0]
        # Get the id of the image
        nbr = int(splittedName[1])
        # Detect the face in the image
        faces = faceCascade.detectMultiScale(image)
        # If face is detected, append the face to images and the id to ids
        for (x, y, w, h) in faces:
            images.append(image[y: y + h, x: x + w])
            # print("{} = {}".format(name, nbr))
            ids.append(nbr)
            names.append(name)

            height = int(np.size(image, 0) * (video_scale / 2))
            width = int(np.size(image, 1) * (video_scale / 2))
            windowName = "Ajout de visages a l'ensemble d'entrainement"
            cv2.namedWindow(windowName, cv2.WINDOW_OPENGL);
            cv2.resizeWindow(windowName, width, height)
            image = cv2.resize(image, (width, height))
            cv2.imshow(windowName, image)
            cv2.waitKey(10)
            # return the images list and ids list
    return images, ids, names

# Path to dataset (without ending /, so should be "path/to/dataset")
path = sys.argv[1]

# Delete .DS_Store that might be in the picture database
try:
    os.remove(path + "/.DS_Store")
except OSError:
    pass

if(os.path.isfile(path + "/recognizer.xml")):
    recognizer.load(path + "/recognizer.xml")

elif(os.path.isdir(path)):
    # Get images, ids and names out of pictures contained in the DB
    images, ids, names = get_images_and_ids(path)

    # Train the program with DB pictures
    recognizer.train(images, np.array(ids))

    # Save the recognizer for next time
    recognizer.save(path + "/recognizer.xml")

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
    gray,
    # 1.1 default
    scaleFactor=1.3,
    # 5 default
    minNeighbors=5,
    # 30x30 defualt
    minSize=(30, 30),
    flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    # Loop though every faces on the frame
    for (x, y, w, h) in faces:
        # Predict faces
        id_predicted, conf = recognizer.predict(gray[y: y + h, x: x + w])
        # Get the name for the predicated id
        name_predicated = get_name_for_id(id_predicted, path)
        # if conf is included between [0;150]
        if conf <= 150 :
            conf = (150 - conf)*100/150
        else:
            conf = 0

        # Round float number to 2 decimals
        conf = math.ceil(conf*100)/100

        # Blue -> 0% confidence
        # Green -> 100% confidence
        r = 0
        g = conf*255/100
        b = 255-conf*255/100

        # Names and % are scaled proportionally regarding the subject distance from camera
        if(dynamicTextScaling):
            # video_scale doesn't matter here
            widthScreen = int(np.size(frame, 1))
            widthSquare = w
            textScale = (widthSquare*dynamicTextScalingRate/widthScreen)
        else:
            textScale = staticTextScalingSize

        # Write the guessed name over the face aswell as the confidence percentage
        cv2.putText(frame, "{} ({}%)".format(name_predicated, conf), (int(x - textScale/3), y), cv2.FONT_HERSHEY_SIMPLEX, textScale, (b,g,r), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.putText(frame, "Samuel Prevost, Loris Code (c) 2016", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    # Calculate the new window/image dimensions
    height = int(np.size(frame, 0) * video_scale)
    width = int(np.size(frame, 1) * video_scale)
    # Apply them to the picture and the window
    cv2.namedWindow('Video', cv2.WINDOW_OPENGL)
    cv2.resizeWindow('Video', width, height)
    cv2.imshow('Video', cv2.resize(frame, (width, height)))

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
