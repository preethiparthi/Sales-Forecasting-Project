from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

# Home page
@app.route('/')
def home():
    return render_template('index.html')


# Prediction route
@app.route('/get', methods=['GET', 'POST'])
def predict():
    try:
        if request.method == 'POST':
            # Get values from form
            features = [float(x) for x in request.form.values()]
            
            # Make prediction
            prediction = model.predict([features])
            
            return render_template('index.html', prediction_text=f"Predicted Sales: {prediction[0]}")
        
        else:
            return render_template('index.html')

    except Exception as e:
        return f"Error: {str(e)}"


# Run the app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
