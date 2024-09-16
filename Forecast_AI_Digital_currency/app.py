from flask import Flask, request, jsonify, render_template
import requests
import openai
import pandas as pd
from sklearn.linear_model import LinearRegression
from textblob import TextBlob

app = Flask(__name__)

# دریافت اخبار
def get_news(api_key):
    url = f'https://newsapi.org/v2/everything?q=cryptocurrency&apiKey={api_key}'
    response = requests.get(url)
    return response.json()

# دریافت قیمت‌ های ارز دیجیتال
def get_crypto_prices():
    url = 'https://api.coingecko.com/api/v3/coins/markets'
    params = {
        'vs_currency': 'usd',
        'order': 'market_cap_desc',
        'per_page': 100,
        'page': 1,
        'sparkline': False
    }
    response = requests.get(url, params=params)
    return response.json()

# تحلیل داده ها
# sentiment_analyzer = pipeline('sentiment-analysis')

def analyze_sentiments(news_articles):
    sentiments = []
    for article in news_articles:
        title = article.get('title', '')
        description = article.get('description', '')
        text = f"{title}. {description}" if title or description else ''
        blob = TextBlob(text)
        sentiment = 'POSITIVE' if blob.sentiment.polarity > 0 else 'NEGATIVE'
        sentiments.append(sentiment)
    return sentiments

# ترکیب داده‌ها
def combine_data(prices, sentiments):
    df = pd.DataFrame(prices)
    df['sentiment'] = sentiments
    return df

# پیش‌بینی قیمت‌ها
def predict_prices(df):
    model = LinearRegression()
    X = df[['sentiment']].applymap(lambda x: 1 if x == 'POSITIVE' else 0)
    y = df['current_price']
    model.fit(X, y)
    predictions = model.predict(X)
    return predictions

# تابع تولید متن با استفاده از OpenAI API

def generate_text(predictions, openai_api_key):
    openai.api_key = openai_api_key
    input_text = f"Based on the latest predictions, the price of cryptocurrencies is expected to "
    for i, prediction in enumerate(predictions):
        input_text += f"coin {i+1}: ${prediction:.2f}, "
    input_text = input_text.rstrip(", ") + "."

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": input_text}
        ]
    )
    generated_text = response.choices[0].message['content'].strip()
    return generated_text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    news = get_news('8689e34ee2b64fabaff04f67f5f01269')
    prices = get_crypto_prices()
    sentiments = analyze_sentiments(news['articles'])
    df = combine_data(prices, sentiments)
    predictions = predict_prices(df)
    generated_text = generate_text(predictions, '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')
    return jsonify({'predictions': predictions.tolist(), 'generated_text': generated_text})

if __name__ == '__main__':
    app.run(debug=True)
