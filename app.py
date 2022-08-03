# import section
import pickle
import librosa
import numpy as np
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder , OneHotEncoder

from tensorflow import keras
from tensorflow.keras.models import load_model
import numpy as np
import sys
from keras.models import Sequential
import os
import json

from flask import Flask, render_template, request, redirect
from werkzeug.utils import secure_filename

import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

with open('templates/config.json', 'r') as c:
    parameters = json.load(c) ['parameters']



# Define flask app
app = Flask(__name__)

# some fix variables
app.config['UPLOAD_FOLDER'] = parameters['upload_location']
app.config['FilePath'] = parameters['modelPath']

# Load the model using pickle
loaded_model = pickle.load(open('Models/RTS mlp prediction.h5', 'rb'))

# create predict function
def extract_features_and_predict(file):

    encoder=OneHotEncoder()
    labelencoder=LabelEncoder()

    emotions = {'01':'neutral','02':'calm','03':'happy','04':'sad','05':'angry','06':'fear','07':'disgust','08':'surprised'}
    observed_emotions=['sad','angry','happy','disgust','surprised','neutral','calm','fear']

    # here do preprocess and predict the result
    audioFile = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file))
    
    X, sample_rate = librosa.load(audioFile, sr=22050, res_type='kaiser_fast')
    stft=np.abs(librosa.stft(X))
    result=np.array([])

    mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
    result=np.hstack((result, mfccs))

    chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    result=np.hstack((result, chroma))

    mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    result=np.hstack((result, mel))

    emotions=to_categorical(labelencoder.fit_transform(observed_emotions))
    result=result.reshape(1,-1)
    pred_test = loaded_model.predict(result)


    return pred_test[0]

@app.route('/')
def main():
    print("--------------------- You're on main page -------------------")
    return render_template('main.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if (request.method == 'POST'):
        f = request.files['audioInput']
        filename = f.filename
        f.save( os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename)))
        print("----------------- File uploaded and saved completed file path of audio is: "+ filename + " -------------------")

        predicted_emotion = extract_features_and_predict(filename)

        predicted_emotion= predicted_emotion.upper()

        print(predicted_emotion)
        return render_template('result.html', data = predicted_emotion , path = filename)

@app.route('/respond', methods=['POST'])
def respond():
    dir = "static\savedFiles"
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))
        
    return redirect('/')


app.run()
