import requests
import json

# Define the API endpoint
url = 'http://127.0.0.1:5000/predict'

# Define the payload
payload = {
    'patient_folder': 'BraTS2021_00000/Paitent 1'
}

# Send the POST request
response = requests.post(url, json=payload)

# Print the response
print(response.json())