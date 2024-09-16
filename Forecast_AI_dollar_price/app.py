from flask import Flask, render_template
import requests
import datetime
from transformers import pipeline

app = Flask(__name__)

NEWS_API_URL = "https://newsapi.org/v2/everything"
FX_API_URL = "https://api.exchangerate-api.com/v4/latest/USD"


# دریافت اخبار
def get_news():
    params = {
        'q': 'USD',
        'apiKey': '8689e34ee2b64fabaff04f67f5f01269',
        'language': 'en',
        'sortBy': 'relevancy',
        'from': (datetime.datetime.now() - datetime.timedelta(days=1)).strftime('%Y-%m-%d')
    }
    response = requests.get(NEWS_API_URL, params=params)
    news_data = response.json()
    return news_data['articles']

# دریافت قیمت دلار
def get_fx_rate():
    response = requests.get(FX_API_URL)
    fx_data = response.json()
    return fx_data['rates']['USD']

# پردازش اخبار
def process_news(articles):
    nlp = pipeline("sentiment-analysis")
    sentiments = [nlp(article['description'])[0] for article in articles]
    return sentiments

# پیشبینی قیمت دلار
def predict_fx_rate(news_sentiments, current_fx_rate):
    positive_news = sum(1 for sentiment in news_sentiments if sentiment['label'] == 'POSITIVE')
    negative_news = sum(1 for sentiment in news_sentiments if sentiment['label'] == 'NEGATIVE')
    prediction = current_fx_rate + (positive_news - negative_news) * 0.01  # تغییر قیمت بر اساس تعداد اخبار مثبت و منفی
    return prediction

# تولید متن پیشبینی
def generate_forecast_text(prediction):
    generator = pipeline('text-generation', model='gpt-3')
    text = generator(f"The predicted exchange rate for USD is {prediction:.2f}. This is based on the latest news and market trends.")
    return text[0]['generated_text']

@app.route('/')
def index():
    news_articles = get_news()
    current_fx_rate = get_fx_rate()
    news_sentiments = process_news(news_articles)
    prediction = predict_fx_rate(news_sentiments, current_fx_rate)
    forecast_text = generate_forecast_text(prediction)
    return render_template('index.html', current_fx_rate=current_fx_rate, forecast_text=forecast_text)

if __name__ == "__main__":
    app.run(debug=True)
