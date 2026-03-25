from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("sales_model.pkl")

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        date_str = request.form['date']
        date = pd.to_datetime(date_str)
        input_data = pd.DataFrame({
            'year': [date.year],
            'month': [date.month],
            'day': [date.day]
        })
        prediction = model.predict(input_data)[0]
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)