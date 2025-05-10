import os
import MySQLdb
from flask import Flask, session, url_for, redirect, render_template, request, abort, flash
import tensorflow as tf
import base64
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename
import numpy as np
import joblib
import numpy as np
from flask import Flask, redirect, url_for, request, render_template
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
from database import *
from pathlib import Path
import pandas as pd
import joblib
import pickle
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
# Load the trained LSTM model
model_filename = 'lstm_model.pkl'
with open(model_filename, 'rb') as file:
    clf = pickle.load(file)
 

app = Flask(__name__)
app.secret_key = os.urandom(24)
 # Load the pre-trained model
model = load_model('epileptic_seizure_model.h5')
scaler = joblib.load('scaler.pkl')
# Initialize the scaler (you should save the scaler after training)
 

app.config['UPLOAD_FOLDER'] = 'static/uploads'

@app.route("/")
def home():
    return render_template("main.html")
@app.route("/bhome")
def bhome():
    return render_template("bhome.html")
@app.route("/bl")
def bl():
    return render_template("blogin.html")
@app.route("/br")
def br():
    return render_template("breg.html")
@app.route("/log")
def ll():
    return render_template("main.html")
@app.route("/p")
def p():
    return render_template("p.html")
@app.route("/bregister",methods=['POST','GET'])
def signup():
    if request.method=='POST':
        username=request.form['username']
        email=request.form['email']
        password=request.form['password']
        add=request.form['Location']
        ph=request.form['Phone no']
        status = Buyer_reg(username,email,password,add,ph) 
        if status == 1:
            return render_template("blogin.html")
        else:
            return render_template("breg.html",m1="failed")        
    

@app.route("/blogin",methods=['POST','GET'])
def login():
    if request.method=='POST':
        username=request.form['username']
        password=request.form['password']
        status = Buyer_loginact(request.form['username'], request.form['password'])
        print(status)
        if status == 1: 
            session['username'] = request.form['username']                                     
            return render_template("bhome.html", m1="sucess")
        else:
            return render_template("blogin.html", m1="Login Failed")

@app.route("/pre",methods=['POST','GET'])
def pre():
    features=request.form['inputData']
    #text2=predict_from_input(text)
    # Convert the input string into a list of floats
    features_list = [float(x) for x in features.split()]
        
        # Reshape it to match the model's expected input format
    features_array = np.array([features_list])
        
        # Make a prediction using the trained model
    prediction = clf.predict(features_array)
    print(prediction)
    result=''
    if prediction[0] == 1:
        result="Epileptic Seizure Normal Stage is Detected From EEG Signals"
    if prediction[0] == 2:
        result="Epileptic Seizure Pre-Seizure is Stage Detected From EEG Signals"
    if prediction[0] == 3:
        result="Epileptic Seizure Onset Seizure Stage is Detected From EEG Signals"
    if prediction[0] == 4:
        result="Epileptic Seizure Active Seizure Stage is Detected From EEG Signals"
    if prediction[0] == 5:
        result="Epileptic Seizure Post-Seizure Stage is Detected From EEG Signals"

    return render_template("result.html", text=result)

@app.route("/pre1",methods=['POST','GET'])
def pre1():
    features = request.form['inputData']# Input data as a space-separated string        
    # Convert the input string into a list of floats
    features_list = [float(x) for x in features.split()]
        
        # Rescale the input data using the same scaler used during training
    features_scaled = scaler.transform([features_list])
        
        # Reshape the input to match the model's expected input format (1, timesteps, features)
    features_reshaped = features_scaled.reshape(1, 1, len(features_list))
        
        # Make a prediction using the trained model
    prediction = model.predict(features_reshaped)
    print("ppppppppppppppppppppppppppppppppppp")    
    print(prediction)
        # Convert prediction to class label (the class with the highest probability)
    predicted_class = np.argmax(prediction)
    print("dddddddddddddddddddddddddddddddddddddddddddd")
    print(predicted_class)
    result=''
    if prediction[0] == 1:
        result="Epileptic Seizure Normal Stage is Detected From EEG Signals"
    if prediction[0] == 2:
        result="Epileptic Seizure Pre-Seizure Stage is Detected From EEG Signals"
    if prediction[0] == 3:
        result="Epileptic Seizure Onset Seizure Stage is Detected From EEG Signals"
    if prediction[0] == 4:
        result="Epileptic Seizure Active Seizure Stage is Detected From EEG Signals"
    if prediction[0] == 5:
        result="Epileptic Seizure Post-Seizure Stage is Detected From EEG Signals"

    return render_template("result.html", text=result)


if __name__ == "__main__":
    app.run(debug=True)

     
     