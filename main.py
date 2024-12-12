import streamlit as st
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import pickle

port_stem = PorterStemmer()
vectorization = TfidfVectorizer()

vector_form = pickle.load(open('vector.pkl', 'rb'))
load_model = pickle.load(open('model.pkl', 'rb'))

def stemming(content):
    con=re.sub('[^a-zA-Z]', ' ', content)
    con=con.lower()
    con=con.split()
    filtered_words = []
    for word in con:
        if word not in stopwords.words('english'):
            stemmed_word = port_stem.stem(word)    # Apply stemming to the word
            filtered_words.append(stemmed_word)

    con = filtered_words  # Update `con` with the filtered and stemmed words
    con=' '.join(con)
    return con

def faking_news(news):
    news=stemming(news)
    input_data=[news]
    form1_vector=vector_form.transform(input_data)
    prediction=load_model.predict(form1_vector)
    return prediction

st.markdown(
    """
    <style>
    body {
        background: linear-gradient(to right, #ece9e6, #ffffff);
        font-family: 'Arial', sans-serif;
    }
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #4CAF50;
        margin-top: 20px;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #333;
        margin-bottom: 20px;
    }
    .stTextArea > label {
        font-size: 1rem;
        color: #555;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        font-size: 1rem;
        cursor: pointer;
        border-radius: 5px;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    .details-box {
        background-color: #f9f9f9;
        border: 1px solid #ddd;
        padding: 15px;
        border-radius: 8px;
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

if __name__ == '__main__':
    st.markdown("<div class='main-header'>Fake News Detection Application</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Paste the content of a news article below:</div>", unsafe_allow_html=True)

    sentence = st.text_area("Enter your news content here", "", height=200)
    predict_btt = st.button("Make prediction")
    if predict_btt:
        with st.spinner('Analyzing...'):
            prediction_class = faking_news(sentence)  
        print(prediction_class)
        if prediction_class == [0]:
            st.success('This article is RELIABLE. Stay informed!')
            st.balloons()
        elif prediction_class == [1]:
            st.error('This article appears UNRELIABLE ⚠️.')

    st.markdown(
        """
        <div class='details-box'>
        <strong>What does this app do?</strong>
        <p>This application determines if the news content you provided is reliable or unreliable.</p>
        """,
        unsafe_allow_html=True
    )
