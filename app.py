from flask import Flask, request, render_template

from keras.models import load_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import shutil
from keras.preprocessing import image
from keras.models import load_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
import json
import pickle
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.models import Model, load_model
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Input, Dense, Dropout, Embedding, LSTM
from keras.layers.merge import add

#from keras.preprocessing import image

model=load_model('catdog.h5')

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/",methods=["POST"])
def predict():
    if request.method=="POST":
        '''img=image.load_image(request.files["image"],target_size=(100,100))
        img=image.img_to_array(img)
        img=img/255.0
        img=img.reshape(-1,100,100,3)'''
        f = request.files['image']

        # Save the file to ./uploads
        f.save(f.filename)
        img = image.load_img(f.filename, target_size=(100,100))
        img = image.img_to_array(img)
        img=img/255.0
        img=img.reshape(-1,100,100,3)
        x=model.predict_classes(img)
        if(x[0][0]==0):
            y="cat"
        else:
            y="dog"  



    return render_template("index.html",y=y)

if __name__ == "__main__":
    app.run(debug=True)