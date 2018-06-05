from flask import Flask
from flask import request
from flask import jsonify
from sklearn import svm
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Create a test method
@app.route('/isalive')
def index():
	return "This API is Alive"

@app.route('/prediction', methods=['POST', 'GET'])
def get_prediction():

 # GET the JSONified Pandas dataframe
 print('Requesting...')
 json = request.args.get('data')

 # Transform JSON into Pandas DataFrame
 print('dataframing...')
 df = pd.read_json(json)
 df = df.reset_index(drop=True)

 # Read the serialised model
 print('reading model')
 modelname = 'svm_iris.pkl'
 print('Loading %s' % modelname)
 loaded_model = pickle.load(open(modelname, 'rb'), encoding='latin1')

 # Get predictions
 print('predicting')
 prediction = loaded_model.predict(df)
 prediction_df = pd.DataFrame(prediction)
 prediction_df.columns = ['Species']
 prediction_df.reset_index(drop=True)

 # OPTIONAL: Concatenate Predictions with original Dataframe
 df_with_preds = pd.concat([df, prediction_df], axis=1)
 return df_with_preds.to_json()

if __name__ == '__main__':
 app.run(port=5000,host='0.0.0.0')
 #app.run(debug=True)
