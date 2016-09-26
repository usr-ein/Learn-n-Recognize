#!/usr/bin/env python
# -*- coding: utf-8 -*-

# First file editing:
# 20-09-2016 18h12:45 (UTC+1)
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

import cv2
import os, sys
from PIL import Image
import numpy as np
import math
import datetime

# Arg 1: path to save/load database, database must be folder w/ images or xml file name "recognizer.xml" in it
pathToDB = sys.argv[1]

# Arg 2: threshold value to consider picture taken in learning mod as validity
# In percents, good value is about 65 (%)
thresholdValidity = int(sys.argv[2])

# Arg 3: camera to record video with
video_camera_id = int(sys.argv[3])
# Arg 4: video scale factor (2.5 i.e.)
video_scale = float(sys.argv[4])

# Arg 5: Person's name to learn (escape spaces with \ !!)
subject_name = sys.argv[5]
# Arg 6: Person's id associated to name (should be unique)
subject_id = int(sys.argv[6])

# Rate by which the text size in/decrease, 3.3 default
dynamicTextScalingRate = 3.3

# Are we currently in learning ro scanning mode ?
# 1 = scanning
# 2 = learning
# Scanning by default
mode = 1

def GetHaarCascade(pathCascade = "haarcascade_frontalface_default.xml"):
    # Load the face detection cascade from haar cascade file
    # Should detect human faces in general
    return cv2.CascadeClassifier(pathCascade)

def TouchIDFile(id, name, path):
    os.open("{}/{}.{}.face".format(path, name, id), os.O_CREAT)

def GetLBPHFromDB(path, face_haar_cascade, video_scale = 1.5):
    recognizer = cv2.createLBPHFaceRecognizer()

    if os.path.isfile(path + "/recognizer.xml") :
        print("Chargement de LBPH depuis {}".format(path + "/recognizer.xml"))
        recognizer.load(path + "/recognizer.xml")
        return recognizer, True

    elif(os.path.isdir(path)):
        print("Entraînement LBPH d'après les images contenus dans {}".format(path))

        # Append all the absolute image paths in a list image_paths
        image_paths = [os.path.join(path, f) for f in os.listdir(path)]
        image_paths = [x for x in image_paths if "DS_Store" not in x]
        image_paths = [x for x in image_paths if ".face" not in x]
        image_paths = [x for x in image_paths if ".xml" not in x]

        if len(image_paths) <= 0:
            return recognizer, False

        # images will contains face images
        images = []
        # ids will contains the id that is assigned to the image
        ids = []
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
            faces = face_haar_cascade.detectMultiScale(image)
            # If face is detected, append the face to images and the id to ids
            for (x, y, w, h) in faces:
                images.append(image[y: y + h, x: x + w])
                ids.append(nbr)

                TouchIDFile(nbr, name, path)

                height = int(np.size(image, 0) * (video_scale / 2))
                width = int(np.size(image, 1) * (video_scale / 2))
                windowName = "Ajout de visages a l'ensemble d'entrainement"
                cv2.namedWindow(windowName, cv2.WINDOW_OPENGL);
                cv2.resizeWindow(windowName, width, height)
                image = cv2.resize(image, (width, height))
                cv2.imshow(windowName, image)
                cv2.waitKey(10)

        recognizer.train(images, np.array(ids))
        return recognizer, True

def GetNameForID(id, path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    # image_paths = [x for x in image_paths if "recognizer.xml" not in x]
    # image_paths = [x for x in image_paths if "DS_Store" not in x]
    image_paths = [x for x in image_paths if ".jpg" in x or ".gif" in x or ".png" in x or ".bmp" in x or ".tiff" in x or ".jpeg" in x or ".face" in x]

    name = "[ Name Not Found ]"
    for image_path in image_paths:
        if int(os.path.split(image_path)[1].split(".")[1]) == id :
            name = os.path.split(image_path)[1].split(".")[0]
            break;
        else:
            continue

    return name

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
            if int(os.path.split(face_path)[1].split(".")[1]) == int(id):
                return True
            continue
        except:
            continue

    return False

def SaveDB(path, recognizer):
    print("Sauvegarde de la nouvelle base de donnée, l'ancienne sera renommé en recognizer_{}.xml".format(datetime.datetime.now().isoformat()))

    try:
        os.rename("{}/recognizer.xml".format(path), "{}/recognizer_{}.xml".format(path, datetime.datetime.now().isoformat()))
    except:
        pass

    recognizer.save(path + "/recognizer.xml")

def ConfidenceToPercents(confidence):
    # if conf is included between [0;150]
    if confidence <= 150 :
        confidence = (150 - confidence)*100/150
    else:
        confidence = 0

    # Round float number to 2 decimals
    confidence = math.ceil(confidence*100)/100

    return confidence

def ColorForConfidence(confidence):
    # Blue -> 0% confidence
    # Green -> 100% confidence
    r = 0
    g = conf*255/100
    b = 255-conf*255/100

    return r, g, b

def DynamicTextScale(image, square_width, rate):
    image_width = int(np.size(image, 1))
    textScale = (square_width*rate/image_width)

    return textScale

if IsIDTaken(subject_id, pathToDB) :
    if not query_yes_no("The provided ID is already used by someone, are you sure you wanna continue ?", "no" ):
        exit()

webcam = cv2.VideoCapture(video_camera_id)
faceCascade = GetHaarCascade()
LBPH_recognizer, recognizerStatus = GetLBPHFromDB(pathToDB, faceCascade)

while True:
    # Capture video frame-by-frame
    ret, frame = webcam.read()

    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
    grayFrame,
    # 1.1 default
    scaleFactor=1.3,
    # 5 default
    minNeighbors=5,
    # 30x30 defualt
    minSize=(70, 70),
    flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    # Loop though every faces on the frame
    for (x, y, w, h) in faces:
        # Scanning
        if mode == 1 :
            if not recognizerStatus :
                sys.stdout.write("No image or xml databse found to feed the recognizer, you are about to enter the learning mod, \n Press a key to continue...")
                raw_input()
                mode = 2

            id_predicted, conf = LBPH_recognizer.predict(grayFrame[y: y + h, x: x + w])
            name_predicated = GetNameForID(id_predicted, pathToDB)

            conf = ConfidenceToPercents(conf)

            r, g, b = ColorForConfidence(conf)

            textScale = DynamicTextScale(frame, w, dynamicTextScalingRate)

            # Write the guessed name over the face aswell as the confidence percentage
            cv2.putText(frame, "{} ({}%)".format(name_predicated, conf), (int(x - textScale/3), y), cv2.FONT_HERSHEY_SIMPLEX, textScale, (b,g,r), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            cv2.putText(frame, "Mode scan", (int(np.size(frame, 1)) - 190, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)


        # Learning
        if mode == 2 :
            if recognizerStatus :
                id_predicted, conf = LBPH_recognizer.predict(grayFrame[y: y + h, x: x + w])
            else:
                conf = 151

            conf = ConfidenceToPercents(conf)

            r, g, b = ColorForConfidence(conf)

            textScale = DynamicTextScale(frame, w, dynamicTextScalingRate)

            # Write the subject's name above they (supposed) face
            cv2.putText(frame, "{} ({}%)".format(subject_name, conf), (int(x - textScale/3), y), cv2.FONT_HERSHEY_SIMPLEX, textScale, (b, g, r), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            cv2.putText(frame, "Mode apprentissage", (int(np.size(frame, 1)) - 190, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

            if recognizerStatus :
                if thresholdValidity <= conf:
                    LBPH_recognizer.update([grayFrame[y: y + h, x: x + w]], np.array([subject_id]))
            else:
                LBPH_recognizer.train([grayFrame[y: y + h, x: x + w]], np.array([subject_id]))

    cv2.putText(frame, "Samuel Prevost, Loris Code (c) 2016", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    # Calculate the new window/image dimensions
    height = int(np.size(frame, 0) * video_scale)
    width = int(np.size(frame, 1) * video_scale)
    # Apply them to the picture and the window
    cv2.namedWindow('Video', cv2.WINDOW_OPENGL)
    cv2.resizeWindow('Video', width, height)
    cv2.imshow('Video', cv2.resize(frame, (width, height)))

    # Scanning mode
    if cv2.waitKey(1) & 0xFF == ord('s'):
        if mode == 2 :
            mode = 1
            TouchIDFile(subject_id, subject_name, pathToDB)

    # Learning/training mode
    if cv2.waitKey(1) & 0xFF == ord('l'):
        if mode == 1 :
            mode = 2

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
webcam.release()
cv2.destroyAllWindows()

SaveDB(pathToDB, LBPH_recognizer)
