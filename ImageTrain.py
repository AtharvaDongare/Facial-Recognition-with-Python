import cv2
import numpy as np
from PIL import Image
import os
import PIL
import pickle

#Searching the current directory for the train data images 
BASE_DIR = os.getcwd()
image_dir = os.path.join(BASE_DIR , 'TrainData')

##Opening the haarcascade files and decalring the LPBH classifier 
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

##Initailizing the lists and dictornaries for further use 
current_id = 0
x_train = []
y_labels = []
label_ids = {}

##Iterating through the file names returned by the function
for root , dirs , files in os.walk(image_dir) :
    for file in files :
        ##Searching for images with the specified endings 
        if file.endswith('png') or file.endswith('jpg') or file.endswith('jpeg'):
            path = os.path.join(root , file )
            label = os.path.basename(root)

            ##Storing the names of the folders so that could be further used to label the data
            if label not in label_ids :
                label_ids[label] = current_id
                current_id += 1
            
            ##Converting the image into PIL format and turning it inot grayscale
            id_ = label_ids[label]
            pil_image = Image.open(path).convert('L')

            ##Resizing the images so that better training results could be acquired and storing the same in an np array
            size = (500 ,500)
            final_image = pil_image.resize(size , Image.ANTIALIAS)

            image_array = np.array(final_image , 'uint8')

            ##Detecting the faces in the image and storing the ROI in the array along side the labels 
            faces = face_cascade.detectMultiScale(image_array , scaleFactor= 1.5 , minNeighbors= 2)

            for (x,y,w,h) in faces :
                roi = image_array[y:y+h , x:x+w]
                x_train.append(roi)
                y_labels.append(id_)

## Appending the labels in a binary pickle file 
with open("labels.pickle" , 'wb') as f:
    pickle.dump(label_ids , f) 

##Training the recognizer and generating a yml file out of it 
recognizer.train(x_train , np.array(y_labels))
recognizer.save('trainer.yml')