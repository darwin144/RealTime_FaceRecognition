import cv2
import numpy as np
import os
import playsound


layer_size = np.int32([62500, 32, 16, 8, 4])
#model.setLayerSizes(layer_size)
model = cv2.ml.ANN_MLP_create()
model = cv2.ml.ANN_MLP_load('training5data.yml')
#model.load('testingfix.xml')
face_cas = cv2.CascadeClassifier('C:\Python36\Lib\site-packages\cv2\data\haarcascade_frontalface_alt2.xml')
capture = cv2.VideoCapture(0)
storing_data = True
name = ""

while storing_data:
    detect  = False
    ret, frame = capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cas.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    #gray = cv2.flip(frame, 1)
    encode = [int(cv2.IMWRITE_JPEG_QUALITY), 150]
    result, imgencode = cv2.imencode('.jpg', gray, encode)
    data = np.array(imgencode)
    #LOAD_IMAGE_GRAYSCALE converts the image iinto a 2-D matrix of grayscale.
    #decimg = cv2.imdecode(data, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    decimg = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)

    #test = decimg

    for (x, y, w, h) in faces:
        detect = True
        cv2.rectangle(frame, (x,y+10),(x+w,y+h+20),(255,0,0))
        test = decimg[y+10:y+h+20,x:x+w]
        # r = 112 / test.shape[1]
        dim = (250 , 250)
        test = cv2.resize(test, dim, interpolation = cv2.INTER_AREA)
        unroll = test.reshape(1, 62500).astype(np.float32)

    if detect == True:
        try:
            ret, resp = model.predict(unroll)
            predict = resp.argmax(-1)
        except:
            print ("Unknoww")


        if predict[0] == 0:
            name = "Darwin"

        elif predict[0] == 1:
            name = "Pudan"
        #print "Prediction : Pudan"
        elif predict[0] == 2:
            name = "DarwinBotak"
        #print "Predicted : Tidak diketahui"
        elif predict[0] == 3:
            name = "Bapak"

        elif predict[0] == 4:
            name = "XXX"
        #print "Predicted : Ronaldo"
        cv2.putText(frame, name, (x,y), cv2.FONT_ITALIC, w*0.005, (255, 255, 255))

    cv2.imshow("Hello", frame)


    key = cv2.waitKey(27)
    if key == 27 or key==ord('q'):
        break

cv2.destroyAllWindows()
