# app.py
from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)

# Load your pre-trained model (update the path to your model file)
# The `joblib` library is great for saving and loading scikit-learn models.
model = joblib.load('models/placement_prediction_model.pkl')

@app.route('/')
def home():
    """Renders the main home page with the input form."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles the prediction request from the UI."""
    # Get the data from the form
    data = request.json
    
    # Process the data and make a prediction using your model
    # Note: You'll need to adapt this part to match the features your model expects
    # Example:
    features = [[data['feature1'], data['feature2'], ...]]
    prediction = model.predict(features)[0]
    
    # You might also want to predict salary, depending on your model
    # salary_model = joblib.load('models/salary_prediction_model.pkl')
    # salary_prediction = salary_model.predict(features)[0]
    
    # Return the prediction result as JSON
    return jsonify({
        'placement_prediction': 'Placed' if prediction == 1 else 'Not Placed'
        # 'salary_prediction': salary_prediction
    })

if __name__ == '__main__':
    app.run(debug=True)