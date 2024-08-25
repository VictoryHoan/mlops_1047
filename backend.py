from flask import Flask, request, jsonify, render_template
import pandas as pd
from pycaret.classification import load_model, predict_model

# Initialize the Flask application
app = Flask(__name__)

# Load the trained model
model = load_model('mushroom_classification_model')

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the JSON data from the request
        data = request.get_json(force=True)
        print("Data received from frontend:", data)  # Log the data received

        # Convert JSON data into a DataFrame
        data_unseen = pd.DataFrame([data])
        print("Data converted for prediction:", data_unseen)  # Log the data conversion

        # Check if the data matches expected features
        print("Columns in the data frame:", data_unseen.columns.tolist())

        # Generate prediction using the loaded model
        prediction = predict_model(model, data=data_unseen)
        print("Prediction output:", prediction)  # Log the prediction

        # Extract the predicted label from the prediction DataFrame
        output = prediction['prediction_label'].iloc[0]
        print("Final predicted label:", output)

        return jsonify({'prediction': output})
    
    except Exception as e:
        print("Error occurred:", str(e))  # Log any errors
        return jsonify({"error": str(e)})

# Define a route for the frontend
@app.route('/')
def home():
    return render_template('frontend.html')

if __name__ == '__main__':
    app.run(debug=True)
