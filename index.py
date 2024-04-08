import streamlit as st
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import pandas as pd
import fitz
import pickle
import os
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

stopwords_set = set(stopwords.words('english'))
emoticon_pattern = re.compile('(?::|;|=)(?:-)?(?:\)|\(|D|P)')
headers = {
    'authority': 'www.amazon.com',
    'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
    'accept-language': 'en-US,en;q=0.9,bn;q=0.8',
    'sec-ch-ua': '" Not A;Brand";v="99", "Chromium";v="102", "Google Chrome";v="102"',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36'
}
len_page = 150

# Load the sentiment analysis model and TF-IDF vectorizer
with open('clf.pkl', 'rb') as f:
    clf = pickle.load(f)
with open('tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)

def preprocessing(text):
    text = re.sub('<[^>]*>', '', text)
    emojis = emoticon_pattern.findall(text)
    text = re.sub('[\W+]', ' ', text.lower()) + ' '.join(emojis).replace('-', '')
    prter = PorterStemmer()
    text = [prter.stem(word) for word in text.split() if word not in stopwords_set]
    return " ".join(text)

def reviews_html(url, len_page):
    soups = []
    for page_no in range(1, len_page + 1):
        params = {
            'ie': 'UTF8',
            'reviewerType': 'all_reviews',
            'pageNumber': page_no
        }
        response = requests.get(url, headers=headers, params=params)
        soup = BeautifulSoup(response.text, 'lxml')
        soups.append(soup)
    return soups

def get_reviews(html_data):
    data_dicts = []
    boxes = html_data.select('div[data-hook="review"]')
    for box in boxes:
        try:
            name = box.select_one('[class="a-profile-name"]').text.strip()
        except Exception as e:
            name = 'N/A'

        try:
            stars = box.select_one('[data-hook="review-star-rating"]').text.strip().split(' out')[0]
        except Exception as e:
            stars = 'N/A'   

        try:
            title = box.select_one('[data-hook="review-title"]').text.strip()
        except Exception as e:
            title = 'N/A'

        try:
            datetime_str = box.select_one('[data-hook="review-date"]').text.strip().split(' on ')[-1]
            date = datetime.strptime(datetime_str, '%B %d, %Y').strftime("%d/%m/%Y")
        except Exception as e:
            date = 'N/A'

        try:
            description = box.select_one('[data-hook="review-body"]').text.strip()
        except Exception as e:
            description = 'N/A'

        data_dict = {
            'Name' : name,
            'Stars' : stars,
            'Title' : title,
            'Date' : date,
            'Description' : description
        }

        data_dicts.append(data_dict)
    
    return data_dicts

def extract_text_from_pdf(file_path):
    text = ''
    with fitz.open(file_path) as pdf:
        for page_num in range(pdf.page_count):
            page = pdf[page_num]
            text += page.get_text()
    return text

st.title('Sentiment Analysis')

file = st.file_uploader("Upload PDF file")
url = st.text_input('Enter Amazon URL')

if st.button('Analyze'):
    if file:
        filename = 'temp.pdf'
        with open(filename, 'wb') as f:
            f.write(file.read())

        pdf_text = extract_text_from_pdf(filename)
        preprocessed_text = preprocessing(pdf_text)
        text_vector = tfidf.transform([preprocessed_text])
        sentiment = clf.predict(text_vector)[0]

        if sentiment == 1:
            sentiment_result = 'good'
        else:
            sentiment_result = 'bad'

        st.write(f"The product is {sentiment_result}")

    elif url:
        html_datas = reviews_html(url, len_page)

        reviews = []

        for html_data in html_datas:
            review = get_reviews(html_data)
            reviews += review

        df_reviews = pd.DataFrame(reviews)
        df_reviews.to_csv('reviews.csv', index=False)

        csv_file = pd.read_csv("reviews.csv")
        csv_file.to_html("reviews.html")
        pdfkit.from_file('reviews.html', 'reviews.pdf')

        filename = "reviews.pdf"

        pdf_text = extract_text_from_pdf(filename)
        preprocessed_text = preprocessing(pdf_text)
        text_vector = tfidf.transform([preprocessed_text])
        sentiment = clf.predict(text_vector)[0]

        if sentiment == 1:
            sentiment_result = 'good'
        else:
            sentiment_result = 'bad'

        st.write(f"The product is {sentiment_result}")

        st.write("View Reviews PDF: [Download PDF](reviews.pdf)")
        st.write("Positive Reviews: Coming Soon")
        st.write("Negative Reviews: Coming Soon")
