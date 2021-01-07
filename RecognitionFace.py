'''
import numpy as np
import cv2
import os
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
wajah_dir= os.path.join(BASE_DIR,"datawajah")
latih_dir= os.path.join(BASE_DIR,"latihwajah")
'''
wajahDir = 'datawajah'
latihDir = 'latihwajah'
'''

cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)

faceDetector = cv2.CascadeClassifier("C:\Python36\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml")
faceRecognizer = cv2.face.LBPHFaceRecognizer_create()

faceRecognizer.read('training.yml')
font = cv2.FONT_HERSHEY_SIMPLEX

id = 0
name = ['Tidak diketahui','Darwin', 'Nama Lain']

minWidth = 0.1*cam.get(3)
minHeigth = 0.1*cam.get(4)


while True:
    ret, frame = cap.read()
    cv2.flip(frame, 1)#untuk membuat camera vertical

    abuabu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #frame = cv2.resize(frame, None, fx=0.8, fy=0.8, interpolation=cv2.INTER_AREA)#tambahan, interpolation=cv2.INTER_AREA
    faces = faceDetector.detectMultiScale(abuabu, 1.2, 5, minSize=(round(minWidth), round(minHeigth)))

    for (x, y, w, h) in faces:
        frame = cv2.rectangle(frame, (x,y),(x+w, y+h), (0,0,255), 2)
        id, confidence = faceRecognizer.predict(abuabu[y:y+h, x:x+w]) # jika nilai confidencenya 0 artinya cocok sempurna
        if confidence <=50:
            nameIDs = names[id]
            confidenceTxt = ("{0}".format(round(100-confidence)))
        else:
            nameID = names[0]
            confidenceTxt = ("{0}".format(round(100-confidence)))

        cv2.putText(frame,str(nameID),(x+4,y+h-5), font, 1, (255,255,255), 2)
        cv2.putText(frame,str(confidenceTxt),(x+4,y+h-5), font, 1, (255,255,255), 1)


    cv2.imshow('Input', frame)
    #cv2.imshow('Input', abuabu)
    c = cv2.waitKey(1)
    if c == 27 or c ==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
'''
import numpy
import cv2
import pickle


face_cascade = cv2.CascadeClassifier('C:\Python36\Lib\site-packages\cv2\data\haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('C:\Python36\Lib\site-packages\cv2\data\haarcascade_eye.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('training.yml')

label = {"person name":1}
with open("labels.pickle",'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}

cap = cv2.VideoCapture(0)

while (True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for (x, y, h, w) in faces:
        print(x, y, h, w)
        roi_gray = gray[y:y+h, x:x+w]
        roy_color = frame[y:y+h, x:x+w]

        id_, conf = recognizer.predict(roi_gray)
        if conf >=45: #and conf  <=85:
            print(id_)
            print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255, 255, 255)
            stroke = 2
            cv2.putText(frame, name, (x, y), font, 1, color, stroke, cv2.LINE_AA)


        image_item ="1.jpg"
        cv2.imwrite(image_item,roy_color)

        color = (255, 0, 0)
        stroke = 2
        end_core_x = x + w
        end_core_y = y + h
        cv2.rectangle(frame,(x, y), (end_core_x, end_core_y),color, stroke)
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, eh, ew) in eyes:
            cv2.rectangle(roy_color,(ex,ey), (ex+ew, ey+eh),(0, 255, 0), 2)


        #display result frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q' and 'a'):
        break
#when everything done release the captured
cap.release()
cv2.destroyAllWindows()
