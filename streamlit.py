import streamlit as st
import nltk
from nltk.data import find
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
import pandas as pd
import re
import emoji

# Ensure the 'punkt' tokenizer data is downloaded
try:
    find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Initialize stemmer, lemmatizer, and stopwords
stemmer = LancasterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Load model and vectorizer
tfidf_vectorizer_loaded = joblib.load('tfidf_vectorizer.pkl')
model_loaded = joblib.load('model.pkl')

# Helper functions
def remove_html_tags(text):
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

def remove_punc(text):
    return re.sub(r'[^\w\s]', '', text)

def remove_stopwords(text):
    words = text.split()
    return ' '.join(word for word in words if word not in stop_words)

def remove_emoji(text):
    return emoji.replace_emoji(text, replace='')

def apply_stemming(tokens):
    return [stemmer.stem(token) for token in tokens]

def apply_lemmatization(tokens):
    return [lemmatizer.lemmatize(token) for token in tokens]

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

def predict_and_display(reviews):
    # Preprocess the reviews
    preprocessed_reviews = [preprocess_text(review) for review in reviews]
    
    # Transform the preprocessed reviews
    transformed_reviews = tfidf_vectorizer_loaded.transform(preprocessed_reviews)
    
    # Predict sentiment
    predictions = model_loaded.predict(transformed_reviews)
    prediction_proba = model_loaded.predict_proba(transformed_reviews)
    
    # Display results
    for i, review in enumerate(reviews):
        st.write(f"**Review:** {review}")
        st.write(f"**Prediction:** {'Positive' if predictions[i] == 1 else 'Negative'}")
        st.write(f"**Probability:** {prediction_proba[i]}")
        
def main():
    st.title('Sentiment Analysis App')
    
    st.sidebar.title('Options')
    upload_option = st.sidebar.radio('Choose Input Method', ('Enter Review', 'Upload File'))
    
    if upload_option == 'Enter Review':
        user_input = st.text_area("Enter your review:")
        
        if user_input:  # Check if the input is not empty
            reviews = [user_input]  # Treat the user input as a review body
            predict_and_display(reviews)
        else:
            st.error("Please enter a review for prediction.")
    
    elif upload_option == 'Upload File':
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            if 'review_body' in data.columns:
                reviews = data['review_body'].tolist()
                predict_and_display(reviews)
            else:
                st.error("The uploaded CSV file must contain a 'review_body' column.")
                
if __name__ == '__main__':
    main()
