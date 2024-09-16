from flask import Flask, request, jsonify, render_template
import requests
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer

app = Flask(__name__)

# 1. جمع‌ آوری داده‌ ها

def get_news():
    news_api_url = "https://newsapi.org/v2/everything?q=stock market&apiKey=YOUR_NEWS_API_KEY"
    news_response = requests.get(news_api_url)
    return news_response.json()

def get_stock_data():
    stock_api_url = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=IBM&apikey=YOUR_STOCK_API_KEY"
    stock_response = requests.get(stock_api_url)
    return stock_response.json()

# 2. پیش‌ پردازش داده‌ ها

def preprocess_news(news_data):
    articles = news_data['articles']
    clean_articles = []
    for article in articles:
        text = article['content']
        if text:
            text = re.sub(r'\s+', ' ', text)  # حذف فاصله‌های اضافی
            text = re.sub(r'\W', ' ', text)  # حذف کاراکترهای غیرکلمه‌ای
            clean_articles.append(text)
    return clean_articles

def preprocess_stock(stock_data):
    stock_df = pd.DataFrame(stock_data['Time Series (Daily)']).T
    stock_df = stock_df.apply(pd.to_numeric)
    return stock_df

# 3. تحلیل و پیش‌بینی

def predict_stock_prices(stock_df):
    X = stock_df[['1. open', '2. high', '3. low', '4. close', '5. volume']].values
    y = stock_df['4. close'].shift(-1).dropna().values

    X_train, X_test, y_train, y_test = train_test_split(X[:-1], y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    return predictions

# 4. تولید متن و ارائه نتایج

def generate_text(prompt):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(inputs, max_length=100, num_return_sequences=1)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET'])
def predict():
    news_data = get_news()
    stock_data = get_stock_data()
    
    clean_articles = preprocess_news(news_data)
    stock_df = preprocess_stock(stock_data)
    
    predictions = predict_stock_prices(stock_df)
    
    prompt = "Based on recent news and stock market trends, the predicted stock prices for the upcoming days are as follows: "
    analysis_text = generate_text(prompt + str(predictions))
    
    return jsonify({'prediction': analysis_text})

if __name__ == '__main__':
    app.run(debug=True)
