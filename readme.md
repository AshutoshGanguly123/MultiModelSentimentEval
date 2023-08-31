# Real-Time Sentiment Analysis Service using Fine-tuned BERT and Kafka

# Overview
This project implements a data pipeline for real-time sentiment prediction using various pretrained BERT and GPT models. 
The models can be finetuned using the finetune.py on custom dataset
All the available models can be simultaneously trained and tested with model metrics being populated using finetune_all.py
The pipeline is implemented using Flask, Kafka, and the Transformers library.

# Finetuning
1.For finetuning one model just provide the path to dataset and model name 

available model types - model_types = ["roberta", "textattack", "bert", "gpt2", "distilbert"] 
    python fine_tune_from_models.py --model_type roberta --dataset_path path/to/your/dataset.txt 

2.For finetuning all the models and observing their eval metrics to choose the best one use finetune_all.py

# Data Flow

User Message --> Flask API --> Kafka Publisher --> Kafka Topic --> Kafka Consumer --> BERT Model --> API Response

# Prerequisites
- Python 3.x
- Kafka
- Zookeeper
- Flask
- Transformers library

# Setup Instructions

1. Clone the repository
git clone https://github.com/your_repository_url.git

2. Install Dependencies
pip install -r requirements.txt

3. Start Zookeeper
bin/zookeeper-server-start.sh config/zookeeper.properties

4. Start Kafka
bin/kafka-server-start.sh config/server.properties

5. Start Flask App
python3 main.py


## API Usage

Endpoint: /send_message
- Method: POST
- Data Params: {"user_input": "your_text_here"}

Sample cURL Command
curl --header "Content-Type: application/json" --request POST --data '{"user_input":"this is the best food"}' http://0.0.0.0:5000/send_message

# Stress Testing
You can perform stress testing of the kafka queues using the provided Python script.
python3 api_request.py

