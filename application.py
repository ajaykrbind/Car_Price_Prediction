
from flask import Flask, render_template, request
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)

# Load model and data
model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))
car = pd.read_csv("Cleaned Car.csv")

# Route for homepage (GET only)
@app.route('/', methods=['GET','POST'])
def index():
    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    years = sorted(car['year'].unique(), reverse=True)
    fuel_types = car['fuel_type'].unique()

    companies.insert(0, 'Select Company')
    return render_template('index.html', companies=companies, car_models=car_models, years=years, fuel_types=fuel_types)

# Route for prediction (POST)
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        company = request.form.get('company')
        car_model = request.form.get('car_models')
        year = int(request.form.get('year'))
        fuel_type = request.form.get('fuel_type')
        driven = int(request.form.get('kilo_driven'))

        # Logging inputs
        print(f"Inputs: {company}, {car_model}, {year}, {fuel_type}, {driven}")

        # Create input DataFrame
        input_df = pd.DataFrame([[car_model, company, year, driven, fuel_type]],
                                columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'])

        # Prediction
        prediction = model.predict(input_df)

        # Return prediction
        return str(np.round(prediction[0], 2))

    except Exception as e:
        print(" Error during prediction:", e)
        return "Internal Server Error", 500

if __name__ == '__main__':
    app.run(debug=True)




