from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

application = Flask(__name__)
CORS(application)  

loaded_model = joblib.load('knn_model.pkl')
loaded_lr_model = joblib.load('lr_model.pkl')
loaded_ridge_model = joblib.load('ridge_model.pkl')

mini = [1, 40, 34605073, 34605073, 914, 914, 0, 53, 1, 0, 0]
maxi = [97330, 150351394, 3652634665, 3652634665, 65535, 6273, 60999, 60999, 17, 192, 31]


def min_max_normalize(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val)

@application.route('/predict', methods=['POST'])
def predict():
    input_data_json = request.json
    
    if input_data_json is None:
        return jsonify({'error': 'No input data provided'}), 400

    input_data = []
    for key in input_data_json:
        try:
            input_data.append(int(input_data_json[key]))
        except ValueError:
            return jsonify({'error': 'Input data must contain integers only'}), 400

    if len(input_data) != len(mini):
        return jsonify({'error': 'Input data length does not match expected length'}), 400

    normalized_input = [min_max_normalize(input_data[i], mini[i], maxi[i]) for i in range(len(input_data))]
    normalized_input = np.array(normalized_input).reshape(1, -1)

    knn_output = loaded_model.predict(normalized_input)
    lr_output = loaded_lr_model.predict(normalized_input)
    ridge_output = loaded_ridge_model.predict(normalized_input)

    if ridge_output >= 0.5:
        ridge_output = [1]
    else:
        ridge_output = [0]

    return jsonify({
        'knn_output': knn_output.tolist(),
        'lr_output': lr_output.tolist(),
        'ridge_output': ridge_output
    })

if __name__ == '__main__':
    application.run(debug=True, host='0.0.0.0', port=5000)
