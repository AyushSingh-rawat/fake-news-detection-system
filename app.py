import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
import numpy as np

import re
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()



# # Load the ML model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model_fake.pkl', 'rb'))

#preprocessing the text:
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Remove newline characters
    text = re.sub(r'\n', '', text)

    # Tokenize text
    tokens = word_tokenize(text)

    # Initialize stemmer
    ps = PorterStemmer()

    # Filter out non-alphanumeric tokens, remove stopwords, and apply stemming
    processed_tokens = []
    for token in tokens:
        if token.isalnum() and token not in string.punctuation and token not in stopwords.words('english'):
            processed_tokens.append(ps.stem(token))

    # Join processed tokens back into a string
    processed_text = ' '.join(processed_tokens)

    return processed_text

def predict_fake_news(article):
    # Preprocess the article (if needed)
    transformed_text = preprocess_text(article)
    # Example: vectorize the text using tfidf
    vector_input = tfidf.transform([transformed_text])
    

    # Then, use the model for prediction
    prediction = model.predict(vector_input)[0]
    return prediction

def main():
    st.image('img1.png', use_column_width=True, width=None)
    st.title('Fake News Detection')

    # Customizing layout
    st.markdown(
        """
        <style>
            .full-width {
                width: 100%;
            }
            .btn-container {
                display: flex;
                justify-content: center;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Text input for user to enter the article
    article = st.text_area('Enter Article:', '')

    # Check button
    btn_clicked = st.button('Check')

    if btn_clicked and article:
        prediction = predict_fake_news(article)
        st.subheader('Prediction:')
        if prediction == 1:
            st.error('This article is predicted to be fake.')
        else:
            st.success('This article is predicted to be real.')
    elif btn_clicked and not article:
        st.warning('Please enter an article.')

if __name__ == '__main__':
    main()



