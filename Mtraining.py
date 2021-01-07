import numpy as np
import glob
import cv2
import sys

print("\n**************----------***************\n")

X = np.empty((0, 62500))
y = np.empty((0, 5), 'float')
train = glob.glob('latihwajah/trainingdatafix.npz')
#extracting data from the saved .npz files
for i in train:
    with np.load(i) as data:
        print (data.files)
        training = data['training_image']
        train_labels = data['output_array']
    X = np.vstack((X, training))
    y = np.vstack((y, train_labels))

print ('Image Array Shape: ', X.shape)
print ('Label Array Shape: ', y.shape)

e1 = cv2.getTickCount()

model = cv2.ml.ANN_MLP_create()
layer_sizez = np.int32([62500, 32, 16, 8, 5])
model.setLayerSizes(layer_sizez)
model.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP)
model.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM, 2, 1)
model.setBackpropWeightScale(0.001)

#train
for i in range(5):
    #if (i % 100 == 0):
    print("iterasi:", i)
    model.train(np.float32(X), cv2.ml.ROW_SAMPLE, np.float32(y))

e2 = cv2.getTickCount()
time_taken = (e2-e1) / cv2.getTickFrequency()
print ("Time taken to train : ", time_taken)
print("Input: "+str(X))

#Prediction
ret, resp = model.predict(X)
prediction = resp.argmax(-1)
true_labels = y.argmax(-1)
print("actual output: "+str(true_labels))
train_rate = np.mean(prediction == true_labels)
print ("prediction: "+str(prediction))
print ('Train accuracy: ', "{0:.2f}%".format(train_rate * 100))

#save model result training
model.save('training5data.yml')
print("Berhasil")
