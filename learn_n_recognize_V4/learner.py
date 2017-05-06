#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Latest file editing:
# 18-09-2016 18h19:45 (UTC+1)
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

#Arg 1: path to save "recognizer.xml" and .face files in it
#Arg 2: subject name
#Arg 3: subject id
#Arg 4: webcam id
#Arg 5: video scale factor

# Face validity threshold (in percentages)
thresholdValid = 60

dynamicTextScaling = True
# 3.5 default
dynamicTextScalingRate = 3.3
# 1.5 default
staticTextScalingSize = 1.5
# Did we already initialised the face recognizer ?
initialised = False

def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = raw_input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")

def IsIDTaken(id, path):
    face_paths = [os.path.join(path, f) for f in os.listdir(path) if ".face" in f]

    for face_path in face_paths:
        try:
            if(int(os.path.split(face_path)[1].split(".")[1]) == int(id)):
                return True
            continue
        except:
            continue

    return False


# Load the face detection cascade from haar cascade file
# Should detect human faces in general
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

# Registrating video camera
video_capture = cv2.VideoCapture(int(sys.argv[4]))

# Setting the video scale factor (I'm using 2.5 for testing)
video_scale = float(sys.argv[5])

# Create a face recognizer object
# (Here we're using Local Binary Patterns Histograms (LBPH) algoritm)
recognizer = cv2.createLBPHFaceRecognizer()

# Path to dataset (without ending /, so should be "path/to/dataset")
path = sys.argv[1]

if(IsIDTaken(sys.argv[3], path)):
    if(not query_yes_no("The provided ID is already used by someone, are you sure you wanna continue ?", "no")):
        exit()

# Delete .DS_Store that might be in the picture database
try:
    os.remove(path + "/.DS_Store")
except OSError:
    pass

if(os.path.isfile(path + "/recognizer.xml")):
    print("Loading {} in memory".format(path + "/recognizer.xml"))
    recognizer.load(path + "/recognizer.xml")
    initialised = True

images = []
ids = []
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
    minSize=(80, 80),
    flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    # Loop though every faces on the frame
    for (x, y, w, h) in faces:
        if(initialised):
            try:
                # Predict faces
                id_predicted, conf = recognizer.predict(gray[y: y + h, x: x + w])
            except:
                conf = 151
                initialised = False
        else:
            conf = 1

        # Get the name for the subject's id
        name = sys.argv[2]
        id_subject = int(sys.argv[3])
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
        cv2.putText(frame, "{} ({}%)".format(name, conf), (int(x - textScale/3), y), cv2.FONT_HERSHEY_SIMPLEX, textScale, (b,g,r), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        if(conf >= thresholdValid):
            images.append(gray[y: y + h, x: x + w])
            ids.append(id_subject)

        # im = Image.fromarray(np.uint8(cm.gist_earth(myarray)*255))
        # im.save("{}/{}.{}.png".format(path, name, id_subject))

    cv2.putText(frame, "Samuel Prevost, Loris Code (c) 2016", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    # Calculate the new window/image dimensions
    height = int(np.size(frame, 0) * video_scale)
    width = int(np.size(frame, 1) * video_scale)
    # Apply them to the picture and the window
    cv2.namedWindow('Video', cv2.WINDOW_OPENGL)
    cv2.resizeWindow('Video', width, height)
    cv2.imshow('Video', cv2.resize(frame, (width, height)))

    # Re-train if 't' is pressed
    if cv2.waitKey(1) & 0xFF == ord('t'):
        try:
            try:
                # Don't override itself
                recognizer.update(images, np.array(ids))
            except:
                # Do override itslef but should be first
                recognizer.train(images, np.array(ids))
            images = []
            ids = []
            initialised = True
        except:
            pass

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save the new recognizer
recognizer.save(path + "/recognizer.xml")
# Save file that match name and id
os.open("{}/{}.{}.face".format(path, sys.argv[2], sys.argv[3]), os.O_CREAT)
# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
