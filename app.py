import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from simulator.generateData import generate_data
from algorithms.algorithms import select_algorithm
load_dotenv()

app = Flask(__name__)

DLBAD_PYTHON_RESTAPI_ENDPOINT_KEY = os.getenv(
    "DLBAD_PYTHON_RESTAPI_ENDPOINT_KEY")


def validate_request(func):
    def wrapper(*args, **kwargs):
        if not request.headers['x-api-key'] == DLBAD_PYTHON_RESTAPI_ENDPOINT_KEY:
            return jsonify({'error': 'Geçersiz istek'}), 400
        return func(*args, **kwargs)
    return wrapper


@app.route('/', methods=['GET'])
def home_endpoint():
    return 'DLBAD AI API'


@app.route('/event-trigger', methods=['POST'])
@validate_request
def event_trigger():
    data = request.json

    if data['trigger']['name'] == 'RUN_SIMULATOR':
        simulatorId = data.get('event').get('data').get('new').get('id')
        response = generate_data(1000, simulatorId)

        if response != None:
            return '', 200
        else:
            return 'Error', 400
    elif data['trigger']['name'] == 'RUN_ALGORITHM':
        algorithm_results = data.get('event').get('data').get('new')
        response = select_algorithm(algorithm_results)

        if response != None:
            return '', 200
        else:
            return 'Error', 400
    else:
        return 'Not Found', 404


if __name__ == '__main__':
    app.run(debug=True, port=6080)
