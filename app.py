import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import joblib

app = Flask(__name__)
model = joblib.load('rf.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    print(final_features)
    prediction = model.predict(final_features)

    output = prediction[0]
    print(output)
    if output==0:
        output=' THE APPLICANT HAS NO RISK AND ELIGIBLE FOR THE LOAN'
    else:
        output='THE APPLICANT HAS RISK AND NOT ELIGIBLE FOR THE LOAN'
    

    return render_template('index.html', prediction_text='{}'.format(output))

if __name__ == "__main__":
    app.run(host="localhost", port=7000)

