import streamlit as st
import joblib
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import LancasterStemmer
import string

# Download necessary NLTK data (punkt, stopwords, wordnet)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load saved models and vectorizer
model = joblib.load('naive_bayes_model.joblib')
tfidf_vectorizer_loaded = joblib.load('tfidf_vectorizer.joblib')

# Initialize the lemmatizer, stemmer, and stop words
lemmatizer = WordNetLemmatizer()
stemmer = LancasterStemmer()
stop_words = set(stopwords.words('english'))

# Function to preprocess text
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Tokenization
    sentences = sent_tokenize(text)
    tokenized_sentences = [word_tokenize(sentence) for sentence in sentences]

    # Stemming and Lemmatization
    preprocessed_text = []
    for sentence in tokenized_sentences:
        for word in sentence:
            if word not in stop_words:
                # Apply lemmatization followed by stemming
                lemmatized_word = lemmatizer.lemmatize(word)
                stemmed_word = stemmer.stem(lemmatized_word)
                preprocessed_text.append(stemmed_word)

    return ' '.join(preprocessed_text)

# Function for making predictions and displaying results
def predict_and_display(reviews):
    # Preprocess the reviews
    preprocessed_reviews = [preprocess_text(review) for review in reviews]

    # Transform the preprocessed reviews using the vectorizer
    transformed_reviews = tfidf_vectorizer_loaded.transform(preprocessed_reviews)

    # Make predictions using the loaded model
    predictions = model.predict(transformed_reviews)

    # Display results
    for i, review in enumerate(reviews):
        st.write(f"Review {i+1}: {review}")
        st.write(f"Predicted Sentiment: {'Positive' if predictions[i] == 1 else 'Negative'}")

# Main function
def main():
    st.title("Sentiment Analysis App")
    
    st.write("Enter a review to predict its sentiment:")

    # Single review input
    user_input = st.text_input("Enter review text:")

    if user_input:  # Check if the input is not empty
        reviews = [user_input]  # Treat the user input as a review body
        predict_and_display(reviews)  # Single review prediction
    else:
        st.error("Please enter a review for prediction.")

    # File upload option
    uploaded_file = st.file_uploader("Choose a file (CSV format)", type=["csv"])
    
    if uploaded_file is not None:
        import pandas as pd
        df = pd.read_csv(uploaded_file)
        if 'review_body' in df.columns:
            reviews = df['review_body'].tolist()
            predict_and_display(reviews)
        else:
            st.error("The file does not contain the 'review_body' column.")

if __name__ == '__main__':
    main()
