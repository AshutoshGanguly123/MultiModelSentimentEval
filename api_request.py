import requests
import json
from concurrent.futures import ThreadPoolExecutor

def send_test_message():
    print("in send_test_msg")
    url = 'http://0.0.0.0:5000/send_message'
    headers = {'Content-Type': 'application/json'}
    payload = json.dumps({"user_input": "this is the best food"})
    
    response = requests.post(url, headers=headers, data=payload)
    
    if response.status_code == 200:
        print(f"Server response: {response.json()}")
    else:
        print(f"Failed to send message, status code: {response.status_code}")

def stress_test_serial():
    url = 'http://0.0.0.0:5000/send_message'
    headers = {'Content-Type': 'application/json'}
    payload = json.dumps({"user_input": "this is the best food"})
    
    for i in range(1000):
        response = requests.post(url, headers=headers, data=payload)
        if response.status_code == 200:
            print(f"Server response: {response.json()}")
        else:
            print(f"Failed to send message, status code: {response.status_code}")

def stress_test_parallel():
    with ThreadPoolExecutor() as executor:
        executor.map(send_test_message, range(10))


if __name__ == '__main__':
    #send_test_message()
    stress_test_serial()
    #stress_test_parallel()

