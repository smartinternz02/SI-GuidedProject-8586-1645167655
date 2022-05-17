from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)
import requests

# NOTE: you must manually set API_KEY below using information retrieved from your IBM Cloud account.
API_KEY = "JK4XYxPSH81PORZTxhwNLlIyO6K3GBQezjUTQd2slAiK"
token_response = requests.post('https://iam.cloud.ibm.com/identity/token', data={"apikey": API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'})
mltoken = token_response.json()["access_token"]

header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}

@app.route('/')
def home():
    return render_template("indexEA.html")


@app.route('/predict', methods=["POST", "GET"])
def predict():
    input_features = [float(x) for x in request.form.values()]
    total=[input_features]
   # features_value = [np.array(input_features)]

   # features_name = ['Education', 'JobInvolvement', 'JobLevel', 'DailyRate(USD)',
                  #   'MonthlyIncome(USD)', 'NoofCompaniesWorked'
                  #   ,'TotalWorkingYears', 'YearsAtCompany',
                  #   'YearsInCurrentRole', 'YearsSinceLastPromotion',
                   #  'YearsWithCurrentManager', 'TrainingTimesLastYear', 'PerformanceRating']

   # df = pd.DataFrame(features_value, columns=features_name)
    # NOTE: manually define and pass the array(s) of values to be scored in the next line
    payload_scoring = {"input_data": [{"field": [['Attrition','Education','JobInvolvement','JobLevel','DailyRate(USD)','MonthlyIncome(USD)','NoofCompaniesWorked','TotalWorkingYears','YearsAtCompany','YearsInCurrentRole','YearsSinceLastPromotion','YearsWithCurrentManager','TrainingTimesLastYear','PerformanceRating']],"values":total}]}

    response_scoring = requests.post('https://us-south.ml.cloud.ibm.com/ml/v4/deployments/e265c503-1323-4b0f-9587-3819a0181da4/predictions?version=2022-03-07', json=payload_scoring, headers={'Authorization': 'Bearer ' + mltoken})
    print("Scoring response")
    print(response_scoring.json())
    
    predictions = response_scoring.json()
    print(predictions)

    pred = response_scoring.json()

    output = pred['predictions'][0]['values'][0][0]

    return render_template('resultEA.html', prediction_text=output)


if __name__ == "__main__":
    app.run(debug=False)