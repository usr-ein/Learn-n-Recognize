#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2, os, sys
import numpy as np
from PIL import Image

# dataset should be like %%NAME%%.%NBR%.jpg

#Arg 1: dataset
#Arg 2: images to test

# Load the face detection cascade from haar cascade file
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

# Create a face recognizer object
# (Here we're using Local Binary Patterns Histograms algoritm)
recognizer = cv2.createLBPHFaceRecognizer()

def get_name_for_label(label, path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    name = "[ Name Not Found ]"
    for image_path in image_paths:
        # print("Is {} equal {} ?".format(os.path.split(image_path)[1].split(".")[1], label))
        if(int(os.path.split(image_path)[1].split(".")[1]) == label):
            name = os.path.split(image_path)[1].split(".")[0]
            # print("Yep ! So name is {}".format(name))
            break;
        else:
            # print("Nope")
            continue

    return name


def get_images_and_labels(path):
    # Append all the absolute image paths in a list image_paths
    # We will not read the image with the .sad extension in the training set
    # Rather, we will use them to test our accuracy of the training
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    # images will contains face images
    images = []
    # labels will contains the label that is assigned to the image
    labels = []
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
        # Get the label of the image
        nbr = int(splittedName[1])
        # Detect the face in the image
        faces = faceCascade.detectMultiScale(image)
        # If face is detected, append the face to images and the label to labels
        for (x, y, w, h) in faces:
            images.append(image[y: y + h, x: x + w])
            # print("{} = {}".format(name, nbr))
            labels.append(nbr)
            names.append(name)
            windowName = "Ajout de visages a l'ensemble d'entrainement"
            cv2.resizeWindow(windowName, int(w + (0.35) * w), int(h + (0.35) * h))
            # cv2.namedWindow(windowName, cv2.WINDOW_OPENGL);
            cv2.imshow(windowName, image[y: y + h, x: x + w])
            cv2.waitKey(15)
            # return the images list and labels list
    return images, labels, names

# Path to dataset (without ending /, so should be "path/to/dataset")
path = sys.argv[1]
# Path to the folder containing images to check
pathImageToCheck = sys.argv[2]
# Delete .DS_Store that might be in the picture database
try:
    os.remove(path + "/.DS_Store")
    os.remove(pathImageToCheck + "/.DS_Store")
except OSError:
    pass

images, labels, names = get_images_and_labels(path)

# Train the program
recognizer.train(images, np.array(labels))

# Get the path to the image to check
image_paths = [os.path.join(pathImageToCheck, f) for f in os.listdir(pathImageToCheck)]

predict_images = []
predict_names = []

for image_path in image_paths:
    predict_image_pil = Image.open(image_path).convert('L')
    predict_image = np.array(predict_image_pil, 'uint8')
    faces = faceCascade.detectMultiScale(predict_image)
    for (x, y, w, h) in faces:
        nbr_predicted, conf = recognizer.predict(predict_image[y: y + h, x: x + w])
        name_predicated = get_name_for_label(nbr_predicted, path)
        nbr_actual = os.path.split(image_path)[1].split(".")[1]
        name_actual = os.path.split(image_path)[1].split(".")[0]
        if conf <= 100 :
            conf = 100 - conf
        else:
            conf = 0

        print "{} est Correctement Reconnue comme {}({}) avec une précision de {}".format(name_actual, name_predicated, nbr_predicted, conf)
        # cv2.imshow("Reconaissance facial", predict_image[y: y + h, x: x + w])
        predict_images.append(predict_image)
        predict_names.append(name_predicated)

for x in range(0, len(predict_images) - 1):
    cv2.namedWindow("{} - {}".format(predict_names[x], x), cv2.WINDOW_OPENGL);
    # cv2.resizeWindow("{} - {}".format(predict_names[x], x), int(predict_images[x].shape[1] + (0.35) * predict_images[x].shape[1]), int(predict_images[x].shape[0] + (0.35) * predict_images[x].shape[0]))
    # cv2.resizeWindow("{} - {}".format(predict_names[x], x), 1500, 1500)

    cv2.imshow("{} - {}".format(predict_names[x], x), (predict_images[x])[y: y + h, x: x + w])
# In ms
# cv2.waitKey(10000)
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cv2.destroyAllWindows()
