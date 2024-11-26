import requests
from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

# Retrieve the API key from the environment
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found. Please check your .env file.")

# Set up the request headers with the Groq API key
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

# The data to be sent in the POST request
data = {
    "model": "llama3-8b-8192",
    "messages": [
        {
            "role": "user",
            "content": "Explain the importance of fast language models"
        }
    ]
}

# The URL for the Groq API endpoint
url = "https://api.groq.com/openai/v1/chat/completions"

# Send the POST request to the Groq API
response = requests.post(url, headers=headers, json=data)

# Check the response status code and handle the response
if response.status_code == 200:
    print("Response:", response.json())  # Print the successful response
else:
    print(f"Error {response.status_code}: {response.text}")  # Print the error message
