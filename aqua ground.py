import random
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

import tkinter as tk
from tkinter import scrolledtext
import random
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

# Define the intents dictionary with unique keys for each intent
intents = {
    "Hello": {
        "patterns": ["hello", "good morning"],
        "responses": ["Hello! How can I assist you with groundwater predictions?"]
    },
    "find groundwater": {
        "patterns": ["find groundwater","find ground water at my location."],
        "responses": ["To give you details related to groundwater, please provide the location details."]
    },
    "hosur": {
        "patterns": ["hosur"],
        "responses": [ "Net Annual Rainfall:650 mm  , "    " Total ground water: 1264 ha.m  ,  "    "Ground water worthy areas: 133 sq.km  ,  "    "Quality of water: Portable  ,  "        "source: Aquifers  ,  "    "At your current location you can find water at the deapth of 20 to 180 mts.   "  "Use sensors system for more details."]
    },
    "shoolagiri": {
        "patterns": ["shoolagiri"],
        "responses": ["Net Annual Rainfall: 544 mm  , "    " Total ground water: 1064 ha.m  " "Ground water worthy areas: 241.44 sq.km  ,  "    "Quality of water: Portable  ,  "   "source: Aquifers  ,  "    "In shoolagiri you can find water at the depth of 15 to 160 mts.  "  " Use sensors system for more details."]
    },
    "bargur": {
        "patterns": ["bargur"],
        "responses": ["Net Annual Rainfall:650 mm  , "    " Total ground water: 1264 ha.m  ,  "    "Ground water worthy areas: 133 sq.km  ,  "    "Quality of water: Portable at certain level  ,  "        "source: Aquifers  ,  "    "In bargur you can find water at the deapth of 09 to 150 mts.  "  " Use sensor system for more details.  "]
    },
    "water extraction": {
        "patterns": ["find water discharge at my location","find water quality."],
        "responses": ["Please use the sensor baised system. Thank you"]
    }
}

# Tokenize and preprocess text using NLTK
def tokenize(text):
    tokens = nltk.word_tokenize(text)
    return [token.lower() for token in tokens]

# Prepare data for training the intent classifier
all_patterns = []
intent_labels = []

for intent, data in intents.items():
    for pattern in data["patterns"]:
        all_patterns.append(pattern)
        intent_labels.append(intent)

# Vectorize text data using TF-IDF
vectorizer = TfidfVectorizer(tokenizer=tokenize)
X = vectorizer.fit_transform(all_patterns)

# Train a LinearSVC classifier
classifier = LinearSVC()
classifier.fit(X, intent_labels)

def get_response(user_input):
    # Vectorize the user input using the trained TF-IDF vectorizer
    input_vector = vectorizer.transform([user_input])
    
    # Predict the intent label using the trained classifier
    predicted_intent = classifier.predict(input_vector)[0]
    
    # Retrieve a random response from the matched intent
    if predicted_intent in intents:
        return random.choice(intents[predicted_intent]["responses"])
    else:
        return "Sorry, I'm not sure how to respond to that."

def send_message(event=None):
    user_input = user_entry.get().strip()
    
    if user_input:
        response = get_response(user_input)
        update_conversation(f"You: {user_input}", "user")
        update_conversation(f"AI ASSISTANT: {response}", "assistant")
        
        # Clear user input after processing
        user_entry.delete(0, tk.END)
        conversation_log.see(tk.END)

def update_conversation(message, tag):
    conversation_log.config(state=tk.NORMAL)
    conversation_log.insert(tk.END, message + "\n", tag)
    conversation_log.config(state=tk.DISABLED)

# Create GUI window
root = tk.Tk()
root.title("AI Assistant")

# Create text widget to display conversation
conversation_log = scrolledtext.ScrolledText(root, width=150, height=30, state=tk.DISABLED)
conversation_log.grid(row=0, column=0, columnspan=2, padx=10, pady=10)

# Create entry widget for user input
user_entry = tk.Entry(root, width=50)
user_entry.grid(row=1, column=0, padx=10, pady=10)

# Create send button
send_button = tk.Button(root, text="Send", command=send_message)
send_button.grid(row=1, column=1, padx=10, pady=10)

# Bind enter key to send_message function
root.bind('<Return>', send_message)

# Configure tags for conversation log (differentiate user and assistant messages)
conversation_log.tag_configure("user", foreground="blue")
conversation_log.tag_configure("assistant", foreground="red")

# Start the main event loop
root.mainloop()
