from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)
model = pickle.load(open('EA_model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template("indexEA.html")


@app.route('/predict', methods=["POST", "GET"])
def predict():
    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]

    features_name = ['Education', 'JobInvolvement', 'JobLevel', 'DailyRate(USD)',
                     'MonthlyIncome(USD)', 'NoofCompaniesWorked'
        , 'TotalWorkingYears', 'YearsAtCompany',
                     'YearsInCurrentRole', 'YearsSinceLastPromotion',
                     'YearsWithCurrentManager', 'TrainingTimesLastYear', 'PerformanceRating']

    df = pd.DataFrame(features_value, columns=features_name)
    output = model.predict(df)
    print(output)

    return render_template('resultEA.html', prediction_text=output)


if __name__ == "__main__":
    app.run(debug=False)