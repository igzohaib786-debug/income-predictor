from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        years = request.form.get('feature')
        # We use [0][0] because the model returns a nested array
        prediction = model.predict([[float(years)]])
        result = "{:,.2f}".format(prediction[0][0]) 
        return render_template('index.html', prediction=result)
    except:
        return render_template('index.html', prediction="Error: Enter a valid number")

if __name__ == "__main__":
    app.run(debug=True)