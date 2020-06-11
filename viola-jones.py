from pathlib import Path

import csv
import numpy as np
import cv2
import matplotlib.pyplot as plt

def detect_faces(cascade, image, scaleFactor=1.1, minSize=(30, 30)):
    # Converting to grayscale
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Face detection
    return cascade.detectMultiScale(image_gray, scaleFactor=scaleFactor, minNeighbors=5)

def add_rect_faces(image, rectangles):
    # Let's draw a rectangle around the detected faces
    for (x, y, w, h) in rectangles:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return


# Loading the classifier for frontal face
haar_cascade_frontalface = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

p = Path('Database_real_and_fake_face_160x160')

with open('rectangles.csv', 'w', newline='') as csvfile:
    fieldnames = ['filename', 'x', 'y', 'w', 'h']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for imageFile in list(p.glob('**/*.jpg')):
    
        # Loading the image to be tested in grayscale
        print(imageFile)
        image = cv2.imread(str(imageFile))
        
        # Face detection
        faces_rects = detect_faces(haar_cascade_frontalface, image, scaleFactor=1.1, minSize=(160*0.5, 160*0.5))
        if (len(faces_rects)):
            (x, y, w, h) = faces_rects[0]
            writer.writerow({'filename': imageFile, 'x': x, 'y': y, 'w': w, 'h': h})
        
        # Let's print the no. of faces fond
        print('Faces found: ', len(faces_rects))
        
        # Let's draw a rectangle around the detected faces
        add_rect_faces(image, faces_rects)
        # Convert image to RGB and show image
        cv2.imshow('img', image);
        cv2.waitKey(100)
