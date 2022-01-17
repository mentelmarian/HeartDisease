#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 10:56:37 2022

@author: marian & elisa
"""

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from flask import Flask, request, render_template


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    # GETTING VALUES FROM THE FORM
    features = [x for x in request.form.values()]
    form_data = {
            'Age'  :               [int(features[0])],
            'Sex'   :              [features[1]],    
            'ChestPainType':       [features[2]],
            'RestingBP'    :       [int(features[3])],
            'Cholesterol'   :      [int(features[4])],
            'FastingBS'     :      [int(features[5])],
            'RestingECG'  :        [features[6]],
            'MaxHR'     :          [int(features[7])],
            'ExerciseAngina' :     [features[8]],
            'Oldpeak'    :         [float(features[9])],
            'ST_Slope'   :         [features[10]]
}
    #print(form_data)
    
    # READING THE FILE AND APPLYING THE CLASSIFIER    
    df = pd.read_csv('csv/heart_original.csv')

    X= df.drop('HeartDisease', axis=1)
    y= df['HeartDisease']
    categorical_features_indices = np.where(X.dtypes != np.float)[0]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = CatBoostClassifier(verbose=False,random_state=0,
                              objective= 'CrossEntropy',
        colsample_bylevel= 0.04292240490294766,
        depth= 10,
        boosting_type= 'Plain',
        bootstrap_type= 'MVS')
    
    model.fit(X_train, y_train,cat_features=categorical_features_indices,eval_set=(X_test, y_test))
    model.predict(X_test)
    
    # USING THE CLASSIFIER TO PREDICT OUR DATA
    ddff = pd.DataFrame(form_data)
    y_pred = model.predict(ddff)
    yy_pred = model.predict_proba(ddff)
    print(y_pred)
    percentage_healthy = round(yy_pred[0][0], 3)*100
    percentage_failure = round(yy_pred[0][1], 3)*100
    print(yy_pred)
    
    # RETURNING THE CORRECT TEXT BASED ON THE PREDICTION
    if y_pred == 0:
        return render_template('index.html', prediction_text='THE PATIENT IS {} % NOT LIKELY TO HAVE A HEART FAILURE'.format(percentage_healthy), color_choice = 'green')
    else:
        return render_template('index.html', prediction_text='THE PATIENT IS {} % LIKELY TO HAVE A HEART FAILURE'.format(percentage_failure), color_choice = 'red')

if __name__ == "__main__":
    app.run(debug=False)