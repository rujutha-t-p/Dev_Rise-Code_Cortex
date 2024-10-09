from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)

# Load your pre-trained model
regressor = pickle.load(open('../model/ml_model.pkl', 'rb'))

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    input_data = [
        float(data['CDHI']),
        float(data['CDLO']),
        float(data['DeltaCDW']),
        float(data['DeltaCHW']),
        float(data['GPM']),
        float(data['PresentCHP']),
        float(data['PresentCDS']),
        float(data['PresentCH']),
        float(data['PresentCT']),
        float(data['RT']),
        float(data['Temperature']),
        float(data['WBT_C']),
        float(data['KW_CHH']),
        float(data['KW_TOT']),
    ]
    input_data_reshaped = np.array(input_data).reshape(1, -1)
    prediction = regressor.predict(input_data_reshaped)

    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True, port=3000)
