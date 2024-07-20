import numpy as np
import cv2 as cv

haar_cascade=haar_cascade=cv.CascadeClassifier("C:/Users/jillp/Desktop/Images recognition/HaarCascade(frontalfacedefault).xml")
 
# #Loading features and label arrays
# features=np.load('features.npy')
# labels=np.load('labels.npy')

face_recognizer=cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

people=['ben_afflek','elton_john','jerry_seinfeld','madonna','mindy_kaling']

img=cv.imread(r"C:\Users\jillp\Desktop\Images recognition\ben.jpg")

gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('Person',gray)

##Detecting faces in the image
faces_rect=haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=4)
for (x,y,w,h) in faces_rect:
    faces_roi=gray[y:y+h,x:x+w]

    label,confidence=face_recognizer.predict(faces_roi)
    print(f'Label= {people[label]} with confidence of {confidence}')

    cv.putText(img,str(people[label]), (20,20),cv.FONT_HERSHEY_COMPLEX,1.0,(0,255,0),thickness=2)
    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),thickness=2)
cv.imshow('Detected Image',img)

cv.waitKey(0)
