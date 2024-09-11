import streamlit as st
import joblib
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer
import string
import pandas as pd
import matplotlib.pyplot as plt

# Download necessary NLTK data (punkt, stopwords)
nltk.download('punkt')
nltk.download('punkt_tab')  # Do not remove this as requested
nltk.download('stopwords')

# Load saved models and vectorizer
model = joblib.load('naive_bayes_model.joblib')
tfidf_vectorizer_loaded = joblib.load('tfidf_vectorizer.joblib')

# Initialize the stemmer and stop words
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

    # Stemming
    preprocessed_text = []
    for sentence in tokenized_sentences:
        for word in sentence:
            if word not in stop_words:
                # Apply stemming
                stemmed_word = stemmer.stem(word)
                preprocessed_text.append(stemmed_word)

    return ' '.join(preprocessed_text)

# Function for making predictions and displaying results
def predict_and_display(reviews):
    # Preprocess the reviews
    preprocessed_reviews = [preprocess_text(review) for review in reviews]

    # Transform the preprocessed reviews using the vectorizer (with n-gram)
    transformed_reviews = tfidf_vectorizer_loaded.transform(preprocessed_reviews)

    # Make predictions using the loaded model
    predictions = model.predict(transformed_reviews)

    return predictions

# Function to display the pie chart and optional star rating
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
 
def display_bar_charts(avg_star_rating, num_verified, num_unverified):
    # Create a figure and axis
    fig, ax = plt.subplots(2, 1, figsize=(10, 4))  # Make the entire chart thinner by reducing height

    # Horizontal bar chart for Average Star Rating
    ax[0].barh(['Average Star Rating'], [avg_star_rating], color='skyblue')
    ax[0].set_xlim(0, 5)  # Assuming star ratings are between 0 and 5
    ax[0].set_title('Average Star Rating')
    ax[0].set_xlabel('Rating')

    # Display the value at the end of the bar
    ax[0].text(avg_star_rating + 0.1, 0, f'{avg_star_rating:.2f}', va='center', ha='left', fontsize=12, color='black')

    # Horizontal bar chart for Verified and Unverified Purchases (Verified above Unverified)
    ax[1].barh(['Unverified Purchases', 'Verified Purchases'], [num_unverified, num_verified], color=['red', 'green'])
    ax[1].set_title('Purchase Verification')
    ax[1].set_xlabel('Count')

    # Display the values at the end of each bar
    ax[1].text(num_verified + 0.1, 1, str(num_verified), va='center', ha='left', fontsize=12, color='black')
    ax[1].text(num_unverified + 0.1, 0, str(num_unverified), va='center', ha='left', fontsize=12, color='black')

    # Customize the appearance of the bar charts
    for axis in ax:
        axis.tick_params(axis='x', colors='black')
        axis.tick_params(axis='y', colors='black')
        axis.grid(axis='x', linestyle='--')

    # Adjust the layout to make it visually better
    plt.tight_layout()

    # Display the bar charts using Streamlit
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

                # Check if 'star_rating' column is present
                if 'star_rating' in df.columns:
                    star_ratings = df['star_rating'].tolist()
                    avg_star_rating = sum(star_ratings) / len(star_ratings)
                else:
                    avg_star_rating = None  # No star rating provided

                # Get predictions
                predictions = predict_and_display(reviews)

                # Display the result in a pie chart
                display_pie_chart(predictions)

                # Display the average star rating and purchase information as bar charts
                if avg_star_rating is not None:
                    # Initialize counts for verified and unverified purchases
                    num_verified = 0
                    num_unverified = 0

                    # Check if 'verified_purchase' column is present
                    if 'verified_purchase' in df.columns:
                        verified_purchases = df['verified_purchase'].value_counts()
                        num_verified = verified_purchases.get('Y', 0)
                        num_unverified = verified_purchases.get('N', 0)
                    
                    # Display bar charts
                    display_bar_charts(avg_star_rating, num_verified, num_unverified)
                else:
                    st.write("Average star rating information is not available.")
            else:
                st.error("The file does not contain the 'review_body' column.")

                
if __name__ == '__main__':
    main() 
