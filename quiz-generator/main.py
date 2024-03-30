from flask import Flask
from flask import request
from flask import Response                                #<-CHANGED
import os

import vertexai                                           #<-CHANGED
from vertexai.language_models import TextGenerationModel  #<-CHANGED

# Default quiz settings
TOPIC = "Cloud Computing"
NUM_Q = 10
DIFF = "intermediate"
LANG = "English"

PROMPT = """
Generate a 10-question Multiple-Choice Quiz on Cloud Computing for final year computer science students. 
Cover cloud basics, networking, virtualization, CPU scheduling, crowdsourcing, security, CAP Theorem, and PageRank. 
Include 3 knowledge assessment, 3 conceptual understanding, and 4 application-based questions at an intermediate difficulty level. 
Provide answers in JSON format, with each object containing a question, possible responses, and correct answer.
"""

app = Flask(__name__)  # Create a Flask object.
PORT = os.environ.get("PORT")  # Get PORT setting from environment.
if not PORT:
    PORT = 8080

# Initialize Vertex AI access.
vertexai.init(project="quizgenerator-418708", location="asia-southeast1")  #<-CHANGED
parameters = {                                                 #<-CHANGED
    "candidate_count": 1,                                      #<-CHANGED
    "max_output_tokens": 1024,                                 #<-CHANGED
    "temperature": 0.5,                                        #<-CHANGED
    "top_p": 0.8,                                              #<-CHANGED
    "top_k": 40,                                               #<-CHANGED
}                                                              #<-CHANGED
model = TextGenerationModel.from_pretrained("text-bison")      #<-CHANGED

# This function takes a dictionary, a name, and a default value.
# If the name exists as a key in the dictionary, the corresponding
# value is returned. Otherwise, the default value is returned.
def check(args, name, default):
    if name in args:
        return args[name]
    return default

# The app.route decorator routes any GET requests sent to the /generate
# path to this function, which responds with "Generating:" followed by
# the body of the request.
@app.route("/", methods=["GET"])
# This function generates a quiz using Vertex AI.
def generate():
    args = request.args.to_dict()
    topic = check(args, "topic", TOPIC)
    num_q = check(args, "num_q", NUM_Q)
    diff = check(args, "diff", DIFF)
    prompt = PROMPT.format(topic=topic, num_q=num_q, diff=diff)
    response = model.predict(prompt, **parameters)      #<-CHANGED
    print(f"Response from Model: {response.text}")      #<-CHANGED
    html = f"{response.text}"                           #<-CHANGED
    return Response(html, mimetype="application/json")  #<-CHANGED

# This code ensures that your Flask app is started and listens for
# incoming connections on the local interface and port 8080.
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)