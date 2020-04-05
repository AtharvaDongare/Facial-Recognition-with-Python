import os 
import time 
import cv2
import numpy as np

"""
The program creates a directory by the name which is specified by the user 
The images would be then stored in the newly directory
named from 1 to n 
"""

## The new directory would be named after the name of the candidate
print("Enter the name of the candidate")
name = input()

## Gets the path of the current directory in which we are working 
path = os.getcwd()
path = path +'\\' + name

try : 
    ##Creates the directory with the given name 
    os.mkdir(name)
    print("Directory ", name ," created !")
    i =0 
    ##initializes the counter which would then be used to name the images and maintain the count
    print("Collecting the images for training the data ")
    vid = cv2.VideoCapture(0)
    while True :
        ## Initializes the capturing of video
        check , frame = vid.read()
        i += 1
        
        ## Appends the name of the image type 
        pic_name = str(i) + '.png'

        cv2.imshow('original' , frame)

        ## Saves the image in the specified directory (newly formed)
        cv2.imwrite(path+'\\' +pic_name, frame) 
        key = cv2.waitKey(1)

        ## Exit the recording loop when the character 'q' is encountered 
        if key == ord('q'):
            print("The operation is carried out succesfully ")
            print("The number of images recorded are : " ,i)
            break   
except FileExistsError :

    ## Error message when the file is pre existing 
    print("The file ", name ," already exists")
    print("Its awful u have such a common name")