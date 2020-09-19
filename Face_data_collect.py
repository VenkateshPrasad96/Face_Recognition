# Write a python script that captures images from your webcam video stream
# Extract all faces from the image frame(using haarcascades)
# Store the face information into numpy arrays
 
# 1. Read and show video streams, capture images
# 2. Detect faces and show bounding box
# 3. Flatten the largest face image and save in a numpy array
# 4. Repeat the above for multiple people to generate training data

import cv2
import numpy as np

#initialise camera
cap = cv2.VideoCapture(0)

#Face detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

skip = 0 # parameter to skip every 10th face
face_data = []  #to store the faces
dataset_path = './data/' #to save the data in this path

file_name = input("Enter the name of the person : ")

while True:
    ret,frame = cap.read()

    if ret==False:
            continue #if not able to capture image try again

    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) #to convert image o grayscale to save memory

    faces = face_cascade.detectMultiScale(frame,1.3,5) #to detect the faces
     #to sort faces based on the area(w*h)of the faces, we have used a lambda function implementation to        #calculate area and sort in reverse order
    if len(faces)==0:
        continue
    faces = sorted(faces,key=lambda f:f[2]*f[3])                                                                        

    # Pick the last face(because it tis the largest face acc to area(f[2]*f[3])
    for face in faces[-1:]:
        x,y,w,h = face
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2) # to display a rectangle around the faces
        
        #Extract(Crop out the required face): Region of interest
        offset = 10 #padding of 10 pixels around the face
        face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]
        face_section = cv2.resize(face_section,(100,100))
        
        skip += 1
        if skip%10==0:
            face_data.append(face_section)
            print(len(face_data))
            
    cv2.imshow("Frame",frame)
    cv2.imshow("Face section",face_section)

    key_pressed = cv2.waitKey(1) & 0xFF #gives us an 8 bit integer(32bit & 8 bit)
    if key_pressed == ord('q'): #to generate exit key to close video stream
           break

    
 #Convert our face list array into numpy arrat

face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0],-1))
print(face_data.shape)
    
#Save this data into file system
np.save(dataset_path+file_name+'.npy',face_data)
print("Data Successfully saved at " + dataset_path+file_name+'.npy')
        
cap.release() #releases the video stream
cv2.destroyAllWindows() #closes the video window