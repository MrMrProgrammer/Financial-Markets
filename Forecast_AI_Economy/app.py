from flask import Flask, render_template, request
import requests
import pandas as pd
import openai
from textblob import TextBlob
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

app = Flask(__name__)
openai.api_key = openai_api_key

# جمع‌آوری داده‌ها
def get_news_data(api_key, query, from_date, to_date):
    url = f'https://newsapi.org/v2/everything?q={query}&from={from_date}&to={to_date}&apiKey={api_key}'
    response = requests.get(url)
    return response.json()

def analyze_sentiment(news_data):
    articles = news_data['articles']
    sentiments = []
    for article in articles:
        text = article['description'] if article['description'] else article['title']
        sentiment = TextBlob(text).sentiment.polarity
        sentiments.append(sentiment)
    return sum(sentiments) / len(sentiments) if sentiments else 0

def generate_solutions(economy_forecast):
    prompt = f"براساس تحلیل اخبار اقتصادی و پیش‌بینی وضعیت اقتصادی، راهکارهایی برای بهبود اقتصاد در سال آینده ارائه دهید:\n"
    prompt += f"پیش‌بینی وضعیت اقتصادی به این صورت است که میانگین احساسات {economy_forecast.mean():.2f} است. لطفاً راهکارهای عملی برای بهبود اقتصاد را ارائه دهید."
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=200
    )
    return response.choices[0].text.strip()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/forecast', methods=['POST'])
def forecast():
    today = datetime.today().strftime('%Y-%m-%d')
    last_year = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')

    news_data_economy = get_news_data(news_api_key, 'economy', last_year, today)
    economy_sentiment = analyze_sentiment(news_data_economy)

    economic_indicators = pd.DataFrame({
        'date': pd.date_range(start=last_year, end=today, freq='M'),
        'sentiment': [economy_sentiment for _ in range(12)]
    })

    model = LinearRegression()
    X = economic_indicators.index.factorize()[0].reshape(-1, 1)
    y = economic_indicators['sentiment'].values
    model.fit(X, y)

    future_dates = pd.date_range(start=today, periods=12, freq='M')
    future_X = future_dates.factorize()[0].reshape(-1, 1)
    predicted_economy = model.predict(future_X)

    solutions_text = generate_solutions(predicted_economy)
    
    return render_template('index.html', forecast=solutions_text)

if __name__ == '__main__':
    app.run(debug=True)
