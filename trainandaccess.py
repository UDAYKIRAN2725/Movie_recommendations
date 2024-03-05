from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
import numpy as np

# Sample training data
X_train = ["How are you?", "I am fine.", "What's your name?", "My name is ChatBot."]
y_train = ["greeting", "response", "question", "response"]

# Vectorize the text data
# This line creates a object vectorizer for the countVectorizer class
# The second line creates a matrix with xtrain where columns are the words of xtrain and in the rows the every string from xtrain array , frequencies were written under the words that string belongs 
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)

# Train the decision tree classifier
classifier = DecisionTreeClassifier()
# This line creates a decision tree using the above vectorized data 
classifier.fit(X_train_vectorized, y_train)

# Function to classify user input
# when the input is given to this function , it firstly vectorized like above we had and then predicted the type of statement has the user given with the already prepared decision tree
def classify_input(user_input):
    user_input_vectorized = vectorizer.transform([user_input])
    predicted_class = classifier.predict(user_input_vectorized)
    return predicted_class[0]

# Example usage
print("Bot: Hi there! Ask me anything or say 'exit' to quit.")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        print("Bot: Goodbye!")
        break
    predicted_class = classify_input(user_input)
    if predicted_class == 'greeting':
        print("Bot: Hello! How can I help you?")
    elif predicted_class == 'response':
        print("Bot: That's nice!")
    elif predicted_class == 'question':
        print("Bot: I'm a bot. What's your name?")
    else:
        print("Bot: I'm not sure how to respond to that.")
