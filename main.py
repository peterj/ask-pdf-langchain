# # Get the MISTRAL_API_KEY environment variable
import os
# import requests
# import json
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

api_key = os.environ.get('MISTRAL_API_KEY')

if api_key is None:
    print('MISTRAL_API_KEY environment variable not set')
    exit(1)
model = "mistral-tiny"
client = MistralClient(api_key=api_key)

messages = [
    ChatMessage(role="user", content="What is the best French cheese?")
]

# chat_response = client.chat(
#     model=model,
#     messages=messages,
# )
# print(chat_response.choices[0].message.content)

for chunk in client.chat_stream(model=model, messages=messages):
    delta = chunk.choices[0].delta.content
    print(delta)


# def get_chat_completion(msg):
#     url = 'https://api.mistral.ai/v1/chat/completions'
#     headers = {
#       'Content-Type': 'application/json',
#       'Accept': 'application/json',
#       'Authorization': f'Bearer {MISTRAL_API_KEY}'
#       }
#     data = {
#       'model': 'mistral-tiny',
#       'messages':[
#          {
#            "role": "user",
#            "content": msg
#          }
#       ]
#     }
    
#     # Convert the json object into string
#     data = json.dumps(data)
#     response = requests.post(url, headers=headers, data=data)
    
#     # check the error
#     if response.status_code != 200:
#         print(response.json())
#         exit(1)
    
#     return response.json()
  

# def get_embedding(input):
#     url = 'https://api.mistral.ai/v1/embeddings'
#     headers = {
#       'Content-Type': 'application/json',
#       'Accept': 'application/json',
#       'Authorization': f'Bearer {MISTRAL_API_KEY}'
#       }
#     data = {
#       'model': 'mistral-embed',
#       'input': input
#     }
    
#     # Convert the json object into string
#     data = json.dumps(data)
#     response = requests.post(url, headers=headers, data=data)
    
#     # check the error
#     if response.status_code != 200:
#         print(response.json())
#         exit(1)
    
#     return response.json()

# def get_content(completion_response):
#   return completion_response['choices'][0]['message']['content']

# # content = get_content(get_chat_completion("Who is the most renowned Italian painter?"))
# embd = get_embedding(["This is the first one", "and the second one"])
# print(embd)

