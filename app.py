from flask import Flask, render_template, request, url_for
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("best_model.pkl")

# Homepage route
@app.route('/')
def Home():
    return render_template('index.html')

# Prediction route

@app.route('/predict', methods=['POST'])

def predict():
    if request.method == 'POST':
        try:
            #Toa data kutoka kwenye form
            features = [
                float(request.form['GDP_per_Capita']),
                float(request.form['Social_Support']),
                float(request.form['Healthy_Life_Expectancy']),
                float(request.form['Freedom']),
                float(request.form['Generosity']),
                float(request.form['Corruption_Perception']),
                float(request.form['Unemployment_Rate']),
                float(request.form['Education_Index']),
                float(request.form['Population']),
                float(request.form['Urbanization_Rate']),
                float(request.form['Public_Trust']),
                float(request.form['Mental_Health_Index']),
                float(request.form['Income_Inequality']),
                float(request.form['Public_Health_Expenditure']),
                float(request.form['Climate_Index']),
                float(request.form['Work_Life_Balance']),
                float(request.form['Internet_Access']),
                float(request.form['Crime_Rate']),
                float(request.form['Political_Stability']),
                float(request.form['Employment_Rate']),
                float(request.form['Life_Satisfaction'])
            ]

            #Badilisha kuwa numpy na fanya prediction

            final_features = np.array([features])
            prediction = model.predict(final_features)[0]

            return render_template('index.html',
                                    prediction_text=f"Happiness Score InakadiriwaP: {prediction:.2f}")
        except Exception as e:
            return render_template('index.html', prediction_text=f"Error: {e}")
        
if __name__ == '__main__':
    app.run(debug=True)
        