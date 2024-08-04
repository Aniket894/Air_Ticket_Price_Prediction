from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
import pandas as pd
from src.pipeline.prediction_pipeline import PredictionPipeline

app = Flask(__name__)


class PredictionPipeline:
    def __init__(self, model_path, preprocessor_path):
        # Load the pre-trained model and preprocessor
        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)

    def predict(self, input_data):
        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data])
        # Transform the data using the preprocessor
        processed_data = self.preprocessor.transform(input_df)
        # Make predictions using the model
        predictions = self.model.predict(processed_data)
        return predictions.tolist()
    
# Initialize the prediction pipeline with model and preprocessor paths
prediction_pipeline = PredictionPipeline(
    model_path='artifacts/best_model.pkl',
    preprocessor_path='artifacts/preprocessor.pkl'
)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def get_prediction():
    
        input_data = {
            'airline': request.form.get('airline'),
            'source_city': request.form.get('source_city'),
            'departure_time': request.form.get('departure_time'),
            'stops': request.form.get('stops'),
            'arrival_time': request.form.get('arrival_time'),
            'destination_city': request.form.get('destination_city'),
            'class': request.form.get('class'),
            'days_left': int(request.form.get('days_left')),
            'hours': int(request.form.get('hours')),
            'minutes': int(request.form.get('minutes'))
        }

        # Make prediction
        predictions = prediction_pipeline.predict(input_data)
        
        # Format the prediction to one decimal place
        formatted_prediction = np.round(predictions, 1)
    
        return render_template('results.html', predictions=formatted_prediction)
        

  

if __name__ == '__main__':
    app.run(debug=True)
