from flask import Flask, render_template, jsonify
import requests
import pandas as pd
from datetime import datetime
from transformers import pipeline

app = Flask(__name__)

# جمع‌آوری اخبار
def get_news():
    url = f'https://newsapi.org/v2/everything?q=gold&apiKey={news_api_key}'
    response = requests.get(url)
    news_data = response.json()
    return news_data

# جمع‌آوری قیمت طلا
def get_gold_price():
    url = f'https://metals-api.com/api/latest?access_key={gold_api_key}&base=USD&symbols=XAU'
    response = requests.get(url)
    price_data = response.json()
    return price_data

# پردازش اخبار و تحلیل احساسات
def analyze_sentiment(news_data):
    sentiment_analysis = pipeline('sentiment-analysis')
    sentiments = []
    for article in news_data['articles']:
        sentiment = sentiment_analysis(article['description'])
        sentiments.append(sentiment[0])
    return sentiments

# پیشبینی قیمت طلا
def predict_gold_price(historical_prices):
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    X = historical_prices[['date']].values.reshape(-1, 1)
    y = historical_prices['price']
    model.fit(X, y)
    next_day = datetime.now().timestamp() + 86400  # فردا
    prediction = model.predict([[next_day]])
    return prediction[0]

# تولید گزارش
def generate_report(price_prediction, sentiment_summary):
    generator = pipeline('text-generation', model='gpt-4')
    prompt = f"Based on recent news sentiment and historical price trends, the predicted price of gold for tomorrow is {price_prediction:.2f} USD. Summary of recent news sentiment: {sentiment_summary}"
    report = generator(prompt, max_length=150)
    return report[0]['generated_text']

@app.route('/')
def index():
    news_data = get_news()
    gold_price_data = get_gold_price()
    sentiments = analyze_sentiment(news_data)
    sentiment_summary = f"Positive: {sum([1 for s in sentiments if s['label'] == 'POSITIVE'])}, Negative: {sum([1 for s in sentiments if s['label'] == 'NEGATIVE'])}"

    price_prediction = predict_gold_price(historical_prices)
    report = generate_report(price_prediction, sentiment_summary)
    
    return render_template('index.html', report=report)

if __name__ == '__main__':
    app.run(debug=True)
