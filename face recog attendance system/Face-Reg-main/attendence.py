import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = 'Attendence'
image = []
classnames = []
mylist = os.listdir(path)
print(mylist)
for cl in mylist:
    curimg = cv2.imread(f'{path}/{cl}')
    image.append(curimg)
    classnames.append(os.path.splitext(cl)[0])
print(classnames)

def findEncodings(image):
    encodeList = []
    for img in image:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def attendace(name):
    with open('Attendace.csv','r+') as f:
        mydata = f.readlines()
        nameL = []
        for line in mydata:
            entry = line.split(',')
            nameL.append(entry[0])
        if name not in nameL:
            now = datetime.now()
            dt = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dt}')



encodeListKnown = findEncodings(image)
print('Encoding Complete')

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgs = cv2.resize(img,(0,0),None,0.25,0.25)
    imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)

    facecurfraame = face_recognition.face_locations(imgs)
    encodecurframe = face_recognition.face_encodings(imgs,facecurfraame)

    for encodeFace,faceloc in zip(encodecurframe,facecurfraame):
        match = face_recognition.compare_faces(encodeListKnown,encodeFace)
        facedis = face_recognition.face_distance(encodeListKnown,encodeFace)
        print(facedis)
        matchindex = np.argmin(facedis)

        if match[matchindex]:
            name = classnames[matchindex].upper()
            #print(name)
            y1,x2,y2,x1 = faceloc
            y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img, (x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+7,y2-7),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            attendace(name)

    cv2.imshow('webcam',img)
    cv2.waitKey(1)



#faceloca = face_recognition.face_locations(imgTony)[0]
#encodetony = face_recognition.face_encodings(imgTony)[0]
#cv2.rectangle(imgTony,(faceloca[3],faceloca[0]),(faceloca[1],faceloca[2]),(255,0,255),2)

#facelocaT = face_recognition.face_locations(imgTest)[0]
#encodetest = face_recognition.face_encodings(imgTest)[0]
#cv2.rectangle(imgTest,(facelocaT[3],facelocaT[0]),(facelocaT[1],facelocaT[2]),(255,0,255),2)

#result = face_recognition.compare_faces([encodetony],encodetest)
#facedist = face_recognition.face_distance([encodetony],encodetest)