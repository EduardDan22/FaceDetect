import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# Take the list of images from the ImagesTest folder
path = 'Database'
imagesList = []
classNames = []
myList = os.listdir(path)
print(myList)

# Import every image in the path one by one
for cls in myList:
    curImg = cv2.imread(f'{path}/{cls}')
    imagesList.append(curImg)
    classNames.append(os.path.splitext(cls)[0])
print(classNames)

# Converting images to RGB and encoding them
def findEncodings(imagesList):
    encodeList = []
    for img in imagesList:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

# Entry each person into a csv file
def markPerson(name):
    with open('People.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name}, {dtString}')

encodeListKnown = findEncodings(imagesList)
print('Encoding Complete')

# Initialize webcam
cam = cv2.VideoCapture(0)

while True:
    success, img = cam.read()
    imgSmall = cv2.resize(img, (0, 0), None, 0.25, 0.25)  # reduce the size of image

    facesCurFrame = face_recognition.face_locations(imgSmall)
    encodeCurFrame = face_recognition.face_encodings(imgSmall, facesCurFrame)

    # Grabbing encodeFace from encodeCurFrame and faceLoc from facesCurFrame one by one
    for encodeFace, faceLoc in zip(encodeCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        #print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            #print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x1 * 4, y2 * 4, x2 * 4

            cv2.rectangle(img, (x1,y1), (x2, y2), (0,255,0), 2)  # Create rectangle around the face
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1-200, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)  # Write name on the image feed
            # Show result and accuracy on image
            cv2.putText(img, f'{round(faceDis[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1,
                        (0, 0, 255), 2)

            markPerson(name)

    cv2.imshow('Webcam', img)
    cv2.waitKey(1)