import cv2
import numpy as np
import pickle 

labels = {}

##Declaring cascade classifiers and recognizers to use the predeclared xml and yml files 
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')

##Opening pickle file to access the labels of the trained data
with open("labels.pickle" , 'rb') as f:
    labels = pickle.load(f) 
    labels = {v:k for k,v in labels.items()}

##Initiating the video variable 
cap = cv2.VideoCapture(0)

while True :
    ret , frame = cap.read()

    ##Converting the captured images into gray scale images
    gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)

    ##Identifying the faces in the image 
    faces = face_cascade.detectMultiScale(gray , scaleFactor=1.5 , minNeighbors= 5)
    for x,y,w,h in faces :  
        print(x,y,w,h)

        ##Definig the Roi for the faces in the image
        roi_gray = gray[y:y+h , x:x+w]
        roi_color = frame[y:y+h , x:x+w]

        ##Predicting the image from the recognizer and checking the ID for the same 
        id_ , conf = recognizer.predict(roi_gray)

        ##Checking the confidence for the image to be classified in any one of the category 
        if conf >= 55 and conf <= 95:
            
            ##Printing the ID and the name of the category from the pickel file
            print(id_)
            print(labels[id_])

            ##Defining the params and setting up the font in the video 
            font = cv2.FONT_HERSHEY_COMPLEX
            name = labels[id_]
            color = (0 ,0 ,0)
            stroke = 2    
            cv2.putText(frame , name , (x,y), font , 1 ,color , stroke , cv2.LINE_AA)

        
        ##Drawing a rectangle around the face detected in the image
        color = (255 ,0 , 0)
        stroke = 2
        end_coord_x = x+w
        end_coord_y = y+h
        cv2.rectangle(frame , (x,y) , (end_coord_x , end_coord_y) , color , stroke)

    ##Exiting out of the loop 
    cv2.imshow("Image" , frame)
    key = cv2.waitKey(1)
    if key == ord('q') :
        break