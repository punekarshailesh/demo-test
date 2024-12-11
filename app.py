from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('model_rf.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('original.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        # Collect input data
        input_data = {
            'Age': float(request.form['Age']),
            'BusinessTravel': float(request.form['BusinessTravel']),
            'DailyRate': float(request.form['DailyRate']),
            'DistanceFromHome': float(request.form['DistanceFromHome']),
            'Education': float(request.form['Education']),
            'EducationField': float(request.form['EducationField']),
            'EnvironmentSatisfaction': float(request.form['EnvironmentSatisfaction']),
            'Gender': float(request.form['Gender']),
            'HourlyRate': float(request.form['HourlyRate']),
            'JobInvolvement': float(request.form['JobInvolvement']),
            'JobLevel': float(request.form['JobLevel']),
            'JobRole': float(request.form['JobRole']),
            'JobSatisfaction': float(request.form['JobSatisfaction']),
            'MaritalStatus': float(request.form['MaritalStatus']),
            'MonthlyIncome': float(request.form['MonthlyIncome']),
            'NumCompaniesWorked': float(request.form['NumCompaniesWorked']),
            'OverTime': float(request.form['OverTime']),
            'PercentSalaryHike': float(request.form['PercentSalaryHike']),
            'PerformanceRating': float(request.form['PerformanceRating']),
            'RelationshipSatisfaction': float(request.form['RelationshipSatisfaction']),
            'StockOptionLevel': float(request.form['StockOptionLevel']),
            'TotalWorkingYears': float(request.form['TotalWorkingYears']),
            'TrainingTimesLastYear': float(request.form['TrainingTimesLastYear']),
            'WorkLifeBalance': float(request.form['WorkLifeBalance']),
            'YearsAtCompany': float(request.form['YearsAtCompany']),
            'YearsInCurrentRole': float(request.form['YearsInCurrentRole']),
            'YearsSinceLastPromotion': float(request.form['YearsSinceLastPromotion']),
            'YearsWithCurrManager': float(request.form['YearsWithCurrManager']),
            'Department_Human Resources': float(request.form.get('Department_Human Resources', 0)),
            'Department_Research & Development': float(request.form.get('Department_Research & Development', 0)),
            'Department_Sales': float(request.form.get('Department_Sales', 0))
        }

        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data])

        # Make predictions
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0]

        # Map the result
        result = "Attrition" if prediction == 1 else "No Attrition"
        confidence = prediction_proba[1] if result == "Attrition" else prediction_proba[0]

        # Display result
        return render_template(
            'predict.html',
            prediction=result,
            prediction_color='red' if result == "Attrition" else 'green',
            confidence=f"{confidence:.2%}"
        )

if __name__ == '__main__':
    app.run(debug=True)
