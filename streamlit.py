import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import LancasterStemmer, WordNetLemmatizer
import joblib

# Load pre-trained vectorizer and model
vectorizer = joblib.load('tfidf_vectorizer.joblib')
model = joblib.load('naive_bayes_model.joblib')

# Function to preprocess text
def preprocess_text(text):
    # Remove HTML tags
    pattern = re.compile('<.*?>')
    text = pattern.sub('', text)
    
    # Remove punctuation
    punc = string.punctuation
    text = text.translate(str.maketrans('', '', punc))
    
    # Remove emojis
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols, pictograph
        u"\U0001F680-\U0001F6FF"  # transport and map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U00002FC2-\U0001F251"
        "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    text = " ".join(filtered_words)
    
    # Tokenize and preprocess
    sentences = sent_tokenize(text)
    tokenized_sentences = [word_tokenize(sentence) for sentence in sentences]
    
    lancaster_stemmer = LancasterStemmer()
    lemmatizer = WordNetLemmatizer()
    
    def apply_stemming(words):
        return [lancaster_stemmer.stem(word) for word in words]

    def apply_lemmatization(words):
        return [lemmatizer.lemmatize(word) for word in words]
    
    stemmed_sentences = [apply_stemming(sentence) for sentence in tokenized_sentences]
    lemmatized_sentences = [apply_lemmatization(sentence) for sentence in stemmed_sentences]
    
    return ' '.join([word for sublist in lemmatized_sentences for word in sublist])

# Streamlit app
st.title('Sentiment Analysis')

user_input = st.text_area("Enter review text here:")

if st.button('Predict Sentiment'):
    if user_input:
        preprocessed_text = preprocess_text(user_input)
        text_vector = vectorizer.transform([preprocessed_text])
        prediction = model.predict(text_vector)
        sentiment = 'Positive' if prediction[0] == 1 else 'Negative'
        st.write(f'The sentiment of the review is: {sentiment}')
    else:
        st.write('Please enter some text for analysis.')
