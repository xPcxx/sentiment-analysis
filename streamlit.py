import streamlit as st
from joblib import load
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import LancasterStemmer, WordNetLemmatizer

# Ensure NLTK data is downloaded only if necessary
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')
    stop_words = set(stopwords.words('english'))

# Load your Naive Bayes model and TfidfVectorizer
try:
    nb_model_loaded = load('naive_bayes_model.joblib')
    tfidf_vectorizer_loaded = load('tfidf_vectorizer.joblib')
except FileNotFoundError:
    st.error("Model or vectorizer file not found. Ensure both 'naive_bayes_model.joblib' and 'tfidf_vectorizer.joblib' are present.")
    st.stop()

# Initialize Lancaster stemmer and lemmatizer
lancaster_stemmer = LancasterStemmer()
lemmatizer = WordNetLemmatizer()

# Define preprocessing functions
def remove_html_tags(text):
    pattern = re.compile('<.*?>')
    return pattern.sub('', text)

def remove_punc(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def remove_stopwords(text):
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return " ".join(filtered_words)

def remove_emoji(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def apply_stemming(words):
    return [lancaster_stemmer.stem(word) for word in words]

def apply_lemmatization(words):
    return [lemmatizer.lemmatize(word) for word in words]

# Full text preprocessing pipeline
def preprocess_text(text):
    text = text.lower()
    text = remove_html_tags(text)
    text = remove_punc(text)
    text = remove_stopwords(text)
    text = remove_emoji(text)
    
    # Tokenization
    sentences = sent_tokenize(text)
    tokenized_sentences = [word_tokenize(sentence) for sentence in sentences]
    
    # Stemming and Lemmatization
    stemmed_sentences = [apply_stemming(sentence) for sentence in tokenized_sentences]
    lemmatized_sentences = [apply_lemmatization(sentence) for sentence in stemmed_sentences]
    
    return ' '.join([word for sublist in lemmatized_sentences for word in sublist])

# Streamlit application starts here
def main():
    # Title of your web app
    st.title("Amazon Product Review Sentiment Analysis App")

    # Sidebar for navigation
    st.sidebar.title("Options")
    option = st.sidebar.selectbox("Choose how to input data", ["Enter text", "Upload file"])

    if option == "Enter text":
        # Text box for user input
        user_input = st.text_input("Enter a product review to check sentiment:")

        # Predict button
        if st.button('Predict'):
            if user_input:  # Check if the input is not empty
                # Treat the user input as a review body
                reviews = [user_input]
                predict_and_display(reviews)  # Single review prediction
            else:
                st.error("Please enter a review for prediction.")
    else:  # Option to upload file
        uploaded_file = st.file_uploader("Choose a file", type=['txt', 'csv'])
        if uploaded_file is not None:
            if uploaded_file.type == "text/csv" or uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
                if 'review_body' in data.columns:
                    reviews = data['review_body'].tolist()
                else:
                    st.error("'review_body' column not found in the uploaded CSV file.")
                    return
            else:  # Assume text file
                data = pd.read_table(uploaded_file, header=None, names=['review_body'])
                reviews = data['review_body'].tolist()

            # Check if the file has content
            if not data.empty:
                predict_and_display(reviews)  # File-based prediction

def predict_and_display(reviews):
    # Preprocess the reviews
    preprocessed_reviews = [preprocess_text(review) for review in reviews]

    # Transform the preprocessed reviews
    transformed_reviews = tfidf_vectorizer_loaded.transform(preprocessed_reviews)

    # Make predictions
    results = nb_model_loaded.predict(transformed_reviews)

    # Combine the inputs and predictions into a DataFrame
    results_df = pd.DataFrame({
        'Input': reviews,
        'Prediction': ["Positive" if label == 1 else "Negative" for label in results]
    })

    # Tabulate and display the results
    with st.expander("Show/Hide Prediction Table"):
        st.table(results_df)

    # Display histogram of predictions
    st.write("Histogram of Predictions:")
    fig, ax = plt.subplots()
    prediction_counts = pd.Series(results).value_counts().sort_index()
    prediction_counts.index = ["Negative", "Positive"]
    prediction_counts.plot(kind='bar', ax=ax)
    ax.set_title("Number of Positive and Negative Predictions")
    ax.set_xlabel("Category")
    ax.set_ylabel("Count")
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))  # Ensure y-axis has integer ticks
    st.pyplot(fig)

if __name__ == '__main__':
    main()
