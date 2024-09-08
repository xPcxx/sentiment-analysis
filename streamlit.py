import streamlit as st
import joblib
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import LancasterStemmer
import string
import pandas as pd
import matplotlib.pyplot as plt

# Download necessary NLTK data (punkt, stopwords, wordnet)
nltk.download('punkt')
nltk.download('punkt_tab')
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
    text = str(text).lower()

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

    return predictions


import matplotlib.pyplot as plt

# Function to display the pie chart 
def display_pie_chart(predictions):
    # Count positive and negative predictions
    positive_count = sum(predictions)
    negative_count = len(predictions) - positive_count

    # Data for pie chart
    labels = ['Positive', 'Negative']
    sizes = [positive_count, negative_count]
    
    # Colors similar to the ones in your image
    colors = ['#6d8bc5', '#c5cbe1']  # Dark blue for positive, light blue for negative

    # Plot pie chart
    fig, ax = plt.subplots()
    wedges, _, autotexts = ax.pie(
        sizes, 
        labels=None,  # Remove labels from the pie chart segments
        colors=colors, 
        autopct='%1.1f%%',  # Show percentage inside pie chart
        startangle=90,      # Start angle of pie chart
        textprops=dict(color="black")  # Text color for percentages
    )
    
    # Customize percentage text (inside the pie slices)
    for autotext in autotexts:
        autotext.set_size(14)        # Set font size for percentage text
        autotext.set_weight('bold')  # Make percentage text bold

    # Equal aspect ratio ensures that pie chart is drawn as a circle.
    ax.axis('equal')

    # Add a legend below the pie chart, with a small color box for each label
    plt.legend(wedges, labels, title="Sentiment", loc="upper center", bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=2)
    
    # Display the pie chart using Streamlit
    st.pyplot(fig)




# Main function
def main():
    st.title("Product Review Sentiment Analysis")
    
    st.sidebar.title("Options")
    option = st.sidebar.selectbox("Choose how to input data", ["Enter text", "Upload file"])

    if option == "Enter text":
        user_input = st.text_input("Enter review text:")

        # Check if the input is not empty
        if st.button('Predict'):
            if user_input:  # Check if the input is not empty
                reviews = [user_input]  # Treat the user input as a review body
                predictions = predict_and_display(reviews)  # Single review prediction
                
                # Display the result
                for i, review in enumerate(reviews):
                    st.write(f"Review {i+1}: {review}")
                    st.write(f"Predicted Sentiment: {'Positive' if predictions[i] == 1 else 'Negative'}")
            else:
                st.error("Please enter a review for prediction.")
    
    else:
        # File upload option
        uploaded_file = st.file_uploader("Choose a file (CSV format)", type=["csv"])
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            if 'review_body' in df.columns:
                reviews = df['review_body'].tolist()
                predictions = predict_and_display(reviews)
                
                # Display the result in a pie chart
                display_pie_chart(predictions)
            else:
                st.error("The file does not contain the 'review_body' column.")

if __name__ == '__main__':
    main()
