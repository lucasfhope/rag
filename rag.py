import requests
import json

corpus_of_documents = [
    "Take a leisurely walk in the park and enjoy the fresh air.",
    "Visit a local museum and discover something new.",
    "Attend a live music concert and feel the rhythm.",
    "Go for a hike and admire the natural scenery.",
    "Have a picnic with friends and share some laughs.",
    "Explore a new cuisine by dining at an ethnic restaurant.",
    "Take a yoga class and stretch your body and mind.",
    "Join a local sports league and enjoy some friendly competition.",
    "Attend a workshop or lecture on a topic you're interested in.",
    "Visit an amusement park and ride the roller coasters."
]

user_input = "I like to ski"

def jaccard_similarity(query, document):
    query = query.lower().split(" ")
    document = document.lower().split(" ")
    intersection = set(query).intersection(set(document))
    union = set(query).union(set(document))
    return len(intersection) / len(union)

def return_response(query, corpus):
    similarities = []
    for doc in corpus:
        similarity = jaccard_similarity(query, doc)
        similarities.append(similarity)
    return corpus[similarities.index(max(similarities))]

user_prompt = "What is a leisure activity that you like?"

user_input = "I don't like to hike"

relevant_document = return_response(user_input, corpus_of_documents)

full_response = []

# API documentation: https://github.com/jmorganca/ollama/blob/main/docs/api.md

prompt = f"""
You are a bot that makes recommendations for activities. You answer in a well structured sentence and do not include extra information.
You should not repeat the recommended activity but expand on it instead.
This is the recommended activity: {relevant_document}
The user input is: {user_input}
Compile a recommendation to the user based on the recommended activity and the user input.
"""

url = 'http://localhost:11434/api/generate'

data = {
    "model": "llama3",
    "prompt": prompt
}
headers = {'Content-Type': 'application/json'}

# Debugging step: Print request data
print("Request Data:", json.dumps(data, indent=2))

response = requests.post(url, data=json.dumps(data), headers=headers, stream=True)

try:
    if response.status_code != 200:
        print("Error:", response.status_code, response.text)
    else:
        for line in response.iter_lines():
            if line:
                decoded_line = json.loads(line.decode('utf-8'))
                # Debugging step: Print each line of the response
                print("Decoded Line:", decoded_line)
                if 'response' in decoded_line:
                    full_response.append(decoded_line['response'])
finally:
    response.close()

# Debugging step: Print the full response
print('Full Response:', ''.join(full_response))
