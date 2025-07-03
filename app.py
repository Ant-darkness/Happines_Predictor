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
            
            gdp = float(request.form['GDP_per_Capita'])
            Social_Support = float(request.form['Social_Support'])
            Healthy_Life_Expectancy = float(request.form['Healthy_Life_Expectancy'])
            Freedom = float(request.form['Freedom'])
            Generosity = float(request.form['Generosity'])
            Corruption_Perception = float(request.form['Corruption_Perception'])
            Unemployment_Rate = float(request.form['Unemployment_Rate'])
            Education_Index = float(request.form['Education_Index'])
            Population = float(request.form['Population'])
            Urbanization_Rate = float(request.form['Urbanization_Rate'])
            Public_Trust = float(request.form['Public_Trust'])
            Mental_Health_Index = float(request.form['Mental_Health_Index'])
            Income_Inequality = float(request.form['Income_Inequality'])
            Public_Health_Expenditure = float(request.form['Public_Health_Expenditure'])
            Climate_Index = float(request.form['Climate_Index'])
            Work_Life_Balance = float(request.form['Work_Life_Balance'])
            Internet_Access = float(request.form['Internet_Access'])
            Crime_Rate = float(request.form['Crime_Rate'])
            Political_Stability = float(request.form['Political_Stability'])
            Employment_Rate = float(request.form['Employment_Rate'])
            Life_Satisfaction = float(request.form['Life_Satisfaction'])

            features = [[gdp,Social_Support,  Healthy_Life_Expectancy, Freedom, Generosity, Corruption_Perception, Unemployment_Rate, Education_Index, Population, Urbanization_Rate, Public_Trust, Mental_Health_Index, Income_Inequality, Public_Health_Expenditure, Climate_Index, Work_Life_Balance, Internet_Access, Crime_Rate, Political_Stability, Employment_Rate, Life_Satisfaction]]

    
            prediction = model.predict(features)[0]

            if prediction >= 6:
                interpretation = "Full Peace/ Safe"

            elif prediction >= 4:
                interpretation = "Simple Peace"

            else:
                interpretation = "No more  Peace"

            return render_template('results.html',
                                    prediction=round(prediction, 2),
                                    interpretation=interpretation,
                                    inputs={
                                        "GDP per Capita": gdp,
                                        "Freedom": Freedom,
                                        "Social_Support": Social_Support, 
                                        "Healthy_Life_Expectancy": Healthy_Life_Expectancy, 
                                        "Generosity ": Generosity ,
                                        "Corruption_Perception" : Corruption_Perception, 
                                        "Unemployment_Rate ": Unemployment_Rate ,
                                        "Education_Index " :Education_Index ,
                                        "Population ": Population ,
                                        "Urbanization_Rate" : Urbanization_Rate,
                                        "Public_Trust " : Public_Trust ,
                                        "Mental_Health_Index" : Mental_Health_Index,
                                        "Income_Inequality " : Income_Inequality ,
                                        "Public_Health_Expenditure " : Public_Health_Expenditure ,
                                        "Climate_Index " : Climate_Index ,
                                        "Work_Life_Balance ": Work_Life_Balance ,
                                        "Internet_Access " : Internet_Access ,
                                        "Crime_Rate " : Crime_Rate ,
                                        "Political_Stability" : Political_Stability, 
                                        "Employment_Rate " : Employment_Rate ,
                                        "Life_Satisfaction " : Life_Satisfaction 

                                    })
        except Exception as e:
            return  f"Error: {e}"
        
if __name__ == '__main__':
    app.run(debug=True)
        