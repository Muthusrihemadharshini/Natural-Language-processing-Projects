import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt_tab')

# Load the trained model and vectorizer
with open("chatbot_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Optionally load the vectorizer (if you need to use it separately)
with open("vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Preprocess function to clean and tokenize input
def preprocess_text(text):
    text = text.lower()  # Lowercase the input
    tokens = word_tokenize(text)  # Tokenize the text
    stop_words = set(stopwords.words('english'))  # Load stop words
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    return ' '.join(tokens)

# Function to respond based on user input
def chatbot_response(user_input):
    processed_input = preprocess_text(user_input)  # Preprocess the user input
    predicted_response = model.predict([processed_input])  # Predict the response using the model
    return predicted_response[0]

# Streamlit Interface
import streamlit as st

st.title('Chatbot Example')
user_input = st.text_input("You: ")

if user_input:
    response = chatbot_response(user_input)
    st.write(f"Chatbot: {response}")
