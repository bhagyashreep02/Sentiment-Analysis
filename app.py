from flask import Flask, render_template, request, send_file
import os
import pickle
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from werkzeug.utils import secure_filename
import fitz
import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime
import pdfkit

app = Flask(__name__)

stopwords_set = set(stopwords.words('english'))
emoticon_pattern = re.compile('(?::|;|=)(?:-)?(?:\)|\(|D|P)')
headers = {
    'authority': 'www.amazon.com',
    'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
    'accept-language': 'en-US,en;q=0.9,bn;q=0.8',
    'sec-ch-ua': '" Not A;Brand";v="99", "Chromium";v="102", "Google Chrome";v="102"',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36'
}
len_page = 10

# Load the sentiment analysis model and TF-IDF vectorizer
with open('clf.pkl', 'rb') as f:
    clf = pickle.load(f)
with open('tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'pdf'}

def extract_text_from_pdf(file_path):
    text = ''
    with fitz.open(file_path) as pdf:
        for page_num in range(pdf.page_count):
            page = pdf[page_num]
            text += page.get_text()
    return text

def preprocessing(text):
    text = re.sub('<[^>]*>', '', text)
    emojis = emoticon_pattern.findall(text)
    text = re.sub('[\W+]', ' ', text.lower()) + ' '.join(emojis).replace('-', '')
    prter = PorterStemmer()
    text = [prter.stem(word) for word in text.split() if word not in stopwords_set]
    return " ".join(text)

# Extra Data as Html object from amazon Review page
def reviewsHtml(url, len_page):
    
    # Empty List define to store all pages html data
    soups = []
    
    # Loop for gather all 3000 reviews from 300 pages via range
    for page_no in range(1, len_page + 1):
        
        # parameter set as page no to the requests body
        params = {
            'ie': 'UTF8',
            'reviewerType': 'all_reviews',
            'pageNumber': page_no
        }
        
        # Request make for each page
        response = requests.get(url, headers=headers, params=params)
        
        # Save Html object by using BeautifulSoup4 and lxml parser
        soup = BeautifulSoup(response.text, 'lxml')
        
        # Add single Html page data in master soups list
        soups.append(soup)
        
    return soups


# Grab Reviews name, description, date, stars, title from HTML
def getReviews(html_data):

    # Create Empty list to Hold all data
    data_dicts = []
    
    # Select all Reviews BOX html using css selector
    boxes = html_data.select('div[data-hook="review"]')
    
    # Iterate all Reviews BOX 
    for box in boxes:
        
        # Select Name using css selector and cleaning text using strip()
        # If Value is empty define value with 'N/A' for all.
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
            # Convert date str to dd/mm/yyy format
            datetime_str = box.select_one('[data-hook="review-date"]').text.strip().split(' on ')[-1]
            date = datetime.strptime(datetime_str, '%B %d, %Y').strftime("%d/%m/%Y")
        except Exception as e:
            date = 'N/A'

        try:
            description = box.select_one('[data-hook="review-body"]').text.strip()
        except Exception as e:
            description = 'N/A'

        # create Dictionary with al review data 
        data_dict = {
            'Name' : name,
            'Stars' : stars,
            'Title' : title,
            'Date' : date,
            'Description' : description
        }

        # Add Dictionary in master empty List
        data_dicts.append(data_dict)
    
    return data_dicts

def get_review_link(url):
    new_url_parts = url.split("/")
    for i in range(len(new_url_parts)):
        if new_url_parts[i] == "dp":
            new_url_parts[i] = "product-reviews"
            # Insert the additional segment for review parameters
            new_url_parts.insert(i + 2, "ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews")
            num = i+2
            break  # Exit the loop once the modification is done
    
    new_url_parts = new_url_parts[:num+1:]
    # Reconstruct the URL
    new_url = "/".join(new_url_parts)
    
    return new_url

@app.route('/', methods=['GET', 'POST'])
def index():
    sentiment_result = None
    pdf_filename = None
    if request.method == 'POST':
        if 'file' not in request.files and 'url' not in request.form:
            return render_template('index.html', error='No file or URL provided')

        file = request.files['file']
        pdf_url = request.form['url']
        if file.filename == '' and pdf_url == '':
            return render_template('index.html', error='No selected file or URL')

        if pdf_url:
            pdf_url = get_review_link(pdf_url)

            html_datas = reviewsHtml(pdf_url, len_page)

            reviews = []

            for html_data in html_datas:
                # Grab review data
                review = getReviews(html_data)
                
                # add review data in reviews empty list
                reviews += review

            # Create a dataframe with reviews Data
            df_reviews = pd.DataFrame(reviews)

            # Save data
            df_reviews.to_csv('reviews.csv', index=False)

            csv_file = pd.read_csv("reviews.csv")
            csv_file.to_html("reviews.html")
            pdfkit.from_file('reviews.html', 'reviews.pdf')

            filename = "reviews.pdf"
            pdf_filename = filename

            # Extract text from the downloaded PDF file
            pdf_text = extract_text_from_pdf(filename)

            # Preprocess the extracted text
            preprocessed_text = preprocessing(pdf_text)

            # Transform the preprocessed text into a feature vector
            text_vector = tfidf.transform([preprocessed_text])

            # Predict the sentiment
            sentiment = clf.predict(text_vector)[0]

            if sentiment == 1:
                sentiment_result = 'good'
            else:
                sentiment_result = 'bad'

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join('uploads', filename)
            file.save(file_path)

            # Extract text from the uploaded PDF file
            pdf_text = extract_text_from_pdf(file_path)

            # Preprocess the extracted text
            preprocessed_text = preprocessing(pdf_text)

            # Transform the preprocessed text into a feature vector
            text_vector = tfidf.transform([preprocessed_text])

            # Predict the sentiment
            sentiment = clf.predict(text_vector)[0]

            if sentiment == 1:
                sentiment_result = 'good'
            else:
                sentiment_result = 'bad'

            pdf_filename = None

    return render_template('index.html', sentiment_result=sentiment_result, pdf_filename=pdf_filename)

if __name__ == '__main__':
    app.run(debug=True)
