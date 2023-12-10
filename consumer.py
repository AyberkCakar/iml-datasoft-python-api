import os
from dotenv import load_dotenv
import pika
import json
from algorithms.algorithms import select_algorithm

load_dotenv()
RABBITMQ_HOST = os.getenv("RABBITMQ_HOST")
RABBITMQ_DEFAULT_USER = os.getenv("RABBITMQ_DEFAULT_USER")
RABBITMQ_DEFAULT_PASS = os.getenv("RABBITMQ_DEFAULT_PASS")


def callback(ch, method, properties, body):
    algorithm_results = json.loads(body)
    response = select_algorithm(algorithm_results)

    if response is not None:
        print("The algorithm run successfully.")
    else:
        print("There was a problem running the algorithm.", algorithm_results)


def start_consumer():
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(host=RABBITMQ_HOST, credentials=pika.PlainCredentials(
            RABBITMQ_DEFAULT_USER, RABBITMQ_DEFAULT_PASS))
    )

    channel = connection.channel()

    channel.queue_declare(queue='algorithm_queue', durable=True)

    channel.basic_consume(
        queue='algorithm_queue',
        on_message_callback=callback,
        auto_ack=True
    )

    print(' [*] Waiting for messages. To exit press CTRL+C')
    channel.start_consuming()


if __name__ == '__main__':
    start_consumer()
