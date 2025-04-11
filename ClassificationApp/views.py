from django.shortcuts import render
from django.template import RequestContext
from django.contrib import messages
from django.http import HttpResponse
import os
import pickle
from django.core.files.storage import FileSystemStorage
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import *
import io
import base64

from keras.utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential, load_model, Model
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
import keras

global uname

path = "Dataset"
labels = []
X = []
Y = []

for root, dirs, directory in os.walk(path):
    for j in range(len(directory)):
        name = os.path.basename(root)
        if name not in labels:
            labels.append(name.strip())

X = np.load('model/X.txt.npy')
Y = np.load('model/Y.txt.npy')

X = X.astype('float32')
X = X/255

indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
Y = Y[indices]
Y = to_categorical(Y)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2) #split dataset into train and test
cnn_model = Sequential() 
cnn_model.add(Convolution2D(64, (3, 3), input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]), activation = 'relu'))
cnn_model.add(MaxPooling2D(pool_size = (2, 2)))
cnn_model.add(Convolution2D(28, (3, 3), activation = 'relu'))
cnn_model.add(MaxPooling2D(pool_size = (2, 2)))
cnn_model.add(Flatten())
cnn_model.add(Dense(units = 256, activation = 'relu'))
cnn_model.add(Dense(units = y_train.shape[1], activation = 'softmax'))
cnn_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
if os.path.exists("model/cnn_weights.hdf5") == False:
    model_check_point = ModelCheckpoint(filepath='model/cnn_weights.hdf5', verbose = 1, save_best_only = True)
    hist = cnn_model.fit(X_train, y_train, batch_size = 32, epochs = 35, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
    f = open('model/cnn_history.pckl', 'wb')
    pickle.dump(hist.history, f)
    f.close()    
else:
    cnn_model.load_weights("model/cnn_weights.hdf5")
   
predict = cnn_model.predict(X_test)
predict = np.argmax(predict, axis=1)
y_test1 = np.argmax(y_test, axis=1)
a = accuracy_score(y_test1,predict)*100
p = precision_score(y_test1, predict,average='macro') * 100
r = recall_score(y_test1, predict,average='macro') * 100
f = f1_score(y_test1, predict,average='macro') * 100

def predict(filename, cnn_model):
    global labels
    img = cv2.imread(filename)
    img = cv2.resize(img, (32,32))#resize image
    im2arr = np.array(img)
    im2arr = im2arr.reshape(1,32,32,3)
    img = np.asarray(im2arr)
    img = img.astype('float32')
    img = img/255 #normalizing test image
    predict = cnn_model.predict(img)#now using  efficient_model to predict pest
    predict = np.argmax(predict)
    img = cv2.imread(filename)
    img = cv2.resize(img, (600,400))
    cv2.putText(img, 'Pest Classified As : '+labels[predict], (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 0, 0), 2)
    return img, 'Pest Classified As : '+labels[predict]

def predictModelAction(request):
    if request.method == 'POST':
        cnn_model = load_model("model/cnn_weights.hdf5")
        myfile = request.FILES['t1'].read()
        fname = request.FILES['t1'].name
        if os.path.exists("ClassificationApp/static/test.jpg"):
            os.remove("ClassificationApp/static/test.jpg")
        with open("ClassificationApp/static/test.jpg", "wb") as file:
            file.write(myfile)
        file.close()
        img, msg = predict("ClassificationApp/static/test.jpg", cnn_model)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        plt.imshow(img)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        context= {'data': msg, 'img': img_b64}
        return render(request, 'index.html', context) 

def index(request):
    if request.method == 'GET':
        output = "Classification of Pests using Computer Vision CNN Algorithm"
        context= {'data': output}
        return render(request, 'index.html', context)

def predictModel(request):
    if request.method == 'GET':
        return render(request, 'predictModel.html', {})
    
def trainModel(request):
    if request.method == 'GET':
        global a, p, r, f
        output ='<table border=1 align=center width=100%><tr><th><font size="" color="black">Algorithm Name</th><th><font size="" color="black">Accuracy</th><th><font size="" color="black">Precision</th>'
        output+='<th><font size="" color="black">Recall</th><th><font size="" color="black">FSCORE</th></tr>'
        global accuracy, precision, recall, fscore
        algorithms = ['CNN Algorithm']
        output+='<tr><td><font size="" color="black">'+algorithms[0]+'</td><td><font size="" color="black">'+str(a)+'</td><td><font size="" color="black">'+str(p)+'</td><td><font size="" color="black">'+str(r)+'</td><td><font size="" color="black">'+str(f)+'</td></tr>'
        output+= "</table></br>"
        f = open('model/cnn_history.pckl', 'rb')
        data = pickle.load(f)
        f.close()
        accuracy = data['accuracy']
        loss = data['loss']
        plt.figure(figsize=(6,4))
        plt.grid(True)
        plt.xlabel('Training Epoch')
        plt.ylabel('Accuracy/Loss')
        plt.plot(loss, 'ro-', color = 'red')
        plt.plot(accuracy, 'ro-', color = 'green')
        plt.legend(['Loss', 'Accuracy'], loc='upper left')
        plt.title('CNN Training Accuracy & Loss Graph')
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        img_b64 = base64.b64encode(buf.getvalue()).decode()    
        context= {'data': output, 'img': img_b64}
        return render(request, 'index.html', context)

    
