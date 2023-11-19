import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from simulator.generateData import generate_data
import pika
import json

load_dotenv()

app = Flask(__name__)

RABBITMQ_HOST = os.getenv("RABBITMQ_HOST")
RABBITMQ_DEFAULT_USER = os.getenv("RABBITMQ_DEFAULT_USER")
RABBITMQ_DEFAULT_PASS = os.getenv("RABBITMQ_DEFAULT_PASS")
DLBAD_PYTHON_RESTAPI_ENDPOINT_KEY = os.getenv(
    "DLBAD_PYTHON_RESTAPI_ENDPOINT_KEY")


def send_to_queue(data):
    try:
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(host=RABBITMQ_HOST, credentials=pika.PlainCredentials(
                RABBITMQ_DEFAULT_USER, RABBITMQ_DEFAULT_PASS))
        )
    except pika.exceptions.AMQPConnectionError:
        print("Failed to connect to RabbitMQ server.")

    channel = connection.channel()

    channel.queue_declare(queue='algorithm_queue', durable=True)
    channel.basic_publish(
        exchange='',
        routing_key='algorithm_queue',
        body=json.dumps(data),
        properties=pika.BasicProperties(
            delivery_mode=2,
        ))
    connection.close()


def validate_request(func):
    def wrapper(*args, **kwargs):
        if not request.headers['x-api-key'] == DLBAD_PYTHON_RESTAPI_ENDPOINT_KEY:
            return jsonify({'error': 'Invalid request'}), 400
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
        response = generate_data(500, simulatorId)

        if response != None:
            return '', 200
        else:
            return 'Error', 400
    elif data['trigger']['name'] == 'RUN_ALGORITHM':
        algorithm_results = data.get('event').get('data').get('new')

        if algorithm_results:
            send_to_queue(algorithm_results)
            return '', 202
        else:
            return 'Error', 400
    else:
        return 'Not Found', 404


if __name__ == '__main__':
    app.run(debug=True, port=6080)
