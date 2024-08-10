from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define a route for the home page
@app.route('/')
def home():
    return render_template('home.html')

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Extract input data from the form
    age = float(request.form['Age'])
    tb = float(request.form['TB'])
    alkphos = float(request.form['Alkphos'])
    sgpt = float(request.form['Sgpt'])
    sgot = float(request.form['Sgot'])


    # Preprocess input data as necessary (e.g., convert gender to numerical)
    # Example: Convert gender to numerical (0 for male, 1 for female)
 

    # Prepare input data as a NumPy array
    input_data = np.array([[age, tb, alkphos, sgpt, sgot]])

    # Perform prediction using the model
    prediction = model.predict(input_data)

    # Convert prediction to a human-readable format (if necessary)
    # Example: Convert numerical prediction to text label
    prediction_text = 'Liver disease present' if prediction == 1 else 'Liver disease not present'

    # Pass the prediction result to the template
    return render_template('home.html', Prediction_text=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)
