
# load and evaluate a saved model
import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy import loadtxt
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import socket
from time import sleep
# load model
model = load_model('model.h5')
# summarize model.
model.summary()
# load dataset
IMG_BREDTH = 30
IMG_HEIGHT = 60

cam = cv2.VideoCapture(0)

cv2.namedWindow("Smart Camera")

def A():
    img_counter = 0
    while True:
        ret, frame = cam.read()
        cv2.imshow("test", frame)
        if not ret:
            break
#        k = cv2.waitKey(1)
    
#        if k%256 == 27:
#                # ESC pressed
#                print("Escape hit, closing...")
#                break 
    #    elif k%256 == 32:
                # SPACE pressed      
        img_name = "C:\\Users\\vish1\\OneDrive\\Desktop\\{}.jpg".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1
        
        return img_name



def use_model(path):
    
    model = load_model('model.h5')
    pic = plt.imread(path)
    pic = cv2.resize(pic, (IMG_BREDTH, IMG_HEIGHT))
    pic = np.expand_dims(pic, axis=0)
    classes = model.predict_classes(pic)
    
    #code using PIL
    model = load_model("model.h5")
    pic1 = plt.imread(path)
    pic = Image.open(path).resize((IMG_BREDTH, IMG_HEIGHT))
    plt.imshow(pic1)
    if model.predict_classes(np.expand_dims(pic, axis=0)) == 0:
         classes = '1'
    elif model.predict_classes(np.expand_dims(pic, axis=0)) == 1:
        classes =  '0'
    
    return classes


#socket code for transfering data to raspberry pie
    
TCP_IP = '192.168.43.250'
TCP_PORT = 5005
BUFFER_SIZE = 20  # Normally 1024, but we want fast response
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(("", TCP_PORT))
s.listen(1) 
conn, addr = s.accept()
print ('Connection address:', addr)
boolean_control = True
index = 0
while boolean_control:
   data = conn.recv(BUFFER_SIZE)
   if not data: break
   print ("received data:", data)
   conn.send(bytes(str.encode(use_model("{}".format(A())))))# echo
   index +=1
   if(index > 3):
       break
   else:
       TCP_IP = '192.168.43.250'
       TCP_PORT = 5005
       BUFFER_SIZE = 20  # Normally 1024, but we want fast response
       s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
       s.bind(("", TCP_PORT))
       s.listen(1) 
       conn, addr = s.accept()
       
conn.close()
cam.release()
cv2.destroyAllWindows()


#
#print(use_model(r"C:\Users\vish1\OneDrive\Desktop\{}".format(A())))


