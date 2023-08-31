from kafka import KafkaProducer, KafkaConsumer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from flask import Flask, request, jsonify
import threading
import json
import time
import models

lock = threading.Lock()
sentiment_dict = {}

# Initialize Kafka Producer
producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# Initialize Kafka Consumer
consumer = KafkaConsumer(
    'user_input_topic',
    bootstrap_servers='localhost:9092',
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

#initialize model
tokenizer, model = models.model_textattack()


# Function to process messages with RoBERTa
def process_message(message):
    tokens = tokenizer(message, padding=True, truncation=True, return_tensors="pt")
    output = model(**tokens)
    sentiment = torch.argmax(output.logits).item()
    return sentiment

# Function to consume Kafka topic
def consume_topic():
    for message in consumer:
        msg_id = message.value['id']
        sentiment = process_message(message.value['text'])
        with lock:
            sentiment_dict[msg_id] = sentiment

# Initialize Flask App
app = Flask(__name__)

# Define the API endpoint to get user input
@app.route('/send_message', methods=['POST'])
def send_message():
    user_input = request.json['user_input']
    msg_id = str(time.time())
    producer.send('user_input_topic', {'id': msg_id, 'text': user_input})
    
    # Wait for the consumer to process the message
    while True:
        with lock:
            sentiment = sentiment_dict.pop(msg_id, None)
        if sentiment is not None:
            break
        time.sleep(0.1)
    
    return jsonify({'status': 'message sent', 'sentiment': sentiment})

# Run the Flask App and Kafka Consumer in separate threads
if __name__ == '__main__':
    t1 = threading.Thread(target=app.run, kwargs={'host': '0.0.0.0', 'port': 5000})
    t2 = threading.Thread(target=consume_topic)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
