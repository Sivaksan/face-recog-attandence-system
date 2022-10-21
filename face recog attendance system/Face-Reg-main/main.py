import cv2
import numpy as np
import face_recognition

imgTony = face_recognition.load_image_file('imageTest/Tony stark.jpg')
imgTony = cv2.cvtColor(imgTony,cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('imageTest/TonyS.jpg')
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

faceloca = face_recognition.face_locations(imgTony)[0]
encodetony = face_recognition.face_encodings(imgTony)[0]
cv2.rectangle(imgTony,(faceloca[3],faceloca[0]),(faceloca[1],faceloca[2]),(255,0,255),2)

facelocaT = face_recognition.face_locations(imgTest)[0]
encodetest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(facelocaT[3],facelocaT[0]),(facelocaT[1],facelocaT[2]),(255,0,255),2)

result = face_recognition.compare_faces([encodetony],encodetest)
facedist = face_recognition.face_distance([encodetony],encodetest)
print(result,facedist)
cv2.putText(imgTest,f'{result}{round(facedist[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

cv2.imshow('Tony stark',imgTony)
cv2.imshow('Tony Test',imgTest)
cv2.waitKey(0)

