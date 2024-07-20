
###FACE DETECTION AND FACE RECOGNITION(using built-in recognizer and building own)###

#Face Detection=Only detects the presence of the face
#Face Recognition=Identifying whose face it is


#HARCASCADE-we will use "haarcascade_frontalface_default.xml"
# Uses the edges to detect the faces and not the skin tone or other features

# import cv2 as cv
# img=cv.imread("C:/Users/jillp/Desktop/Images recognition/FACE2.jpg")
# cv.imshow('Person 1',img)

# #--1. converting the image to grayscale
# gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# cv.imshow('Gray Person 1',gray1)

# #--2. Reading the Haar cascade file
# haar_cascade=cv.CascadeClassifier("C:/Users/jillp/Desktop/Images recognition/HaarCascade(frontalfacedefault).xml")

# #--3. Detecting the face

# faces_rect=haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=3)
#   #minNeighbors parameter=>specifies the no. of neighbours the rectangles should have to be called a face

# print(f"No. of faces detected in the image={len(faces_rect)}")

# # we can loop over the list and grab the coordinates of those images and draw the rectangle over the detected faces

# for (x,y,w,h) in faces_rect:
#     cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),thickness=2)
#     #(x,y)-->point 1
#     #(x+w,y+h)--> point 2

# cv.imshow('detected faces',img)

# cv.waitKey(0)


##Detection of face with many people=> manipulate scalefactor and minneighbours for better detection
# import cv2 as cv
# img=cv.imread("C:/Users/jillp/Desktop/Images recognition/Group2.jpg")
# cv.imshow('Group 2',img)

# #--1. converting the image to grayscale
# gray1=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# cv.imshow('Gray Group 1',gray1)

# #--2. Reading the Haar cascade file
# haar_cascade=cv.CascadeClassifier("C:/Users/jillp/Desktop/Images recognition/HaarCascade(frontalfacedefault).xml")

# #--3. Detecting the face

# faces_rect=haar_cascade.detectMultiScale(gray1,scaleFactor=1.1,minNeighbors=7)
#   #minNeighbors parameter=>specifies the no. of neighbours the rectangles should have to be called a face

# print(f"No. of faces detected in the image={len(faces_rect)}")

# # we can loop over the list and grab the coordinates of those images and draw the rectangle over the detected faces

# for (x,y,w,h) in faces_rect:
#     cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),thickness=2)
#     #(x,y)-->point 1
#     #(x+w,y+h)--> point 2

# cv.imshow('detected faces',img)

# cv.waitKey(0)



###FACE DETECTION with OpenCV's Built-in recognizer###

import os
import cv2 as cv
import numpy as np

#--1. Creating the list of all the folder names of the people
people=['ben_afflek','elton_john','jerry_seinfeld','madonna','mindy_kaling']

# loop over every folder

# people=[]
# for i in os.listdir(r"C:\Users\jillp\Desktop\Images recognition\IMAGES DATASET\Celebrity faces\train"):
#     people.append(i)
# print(people)


#--2. creating a variable
DIR=r"C:\Users\jillp\Desktop\Images recognition\IMAGES DATASET\Celebrity faces\train"

haar_cascade=cv.CascadeClassifier("C:/Users/jillp/Desktop/Images recognition/HaarCascade(frontalfacedefault).xml")

features=[] #face
labels=[] #whose face does it belong to

#--3. creating a function

def create_train():
    for person in people: 
        #looping over every person in the people list
        path= os.path.join(DIR,person)          
        #going to each folder and grabbing the path of the folder by joining person to DIR
        label=people.index(person)


        for img in os.listdir(path):
            img_path=os.path.join(path,img)

            img_array= cv.imread(img_path)
            gray=cv.cvtColor(img_array,cv.COLOR_BGR2GRAY)
            faces_rect=haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=4)
            for (x,y,w,h) in faces_rect:
                faces_region_of_interest=gray[y:y+h,x:x+w] #cropping out the face in the image
                features.append(faces_region_of_interest)
                labels.append(label)


create_train()
print("Training Done-----------------")
# print(f'Length of the features list:{len(features)}')
# print(f'Length of th labels list:{len(labels)}')

##Converting features list and labels list to numpy arrays
features=np.array(features,dtype='object') ##Image Arrays of the faces
labels=np.array(labels) ##Corresponding image arrays of faces LABELS


face_recognizer=cv.face.LBPHFaceRecognizer_create()

## Train the recognizer on the features list and the labels list
face_recognizer.train(features,labels)

face_recognizer.save('face_trained.yml')
np.save('features.npy',features)
np.save('labels.npy',labels)



###BUILDING A DEEP COMPUTER VISION MODEL###

