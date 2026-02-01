import os
import requests
from dotenv import load_dotenv
from transformers import pipeline

keyword = 'NVDA'
language = 'en'
published = 0

pipe = pipeline("text-classification", model="ProsusAI/finbert")

load_dotenv()

API_KEY = os.getenv('key')
if not API_KEY:
    raise ValueError('Missing API_KEY in environment.')

page = 1
articles = []

while True:
    url = (
        'https://newsapi.org/v2/everything?'
        f'q={keyword}&'
        f'language={language}&'
        f'page={page}&'
        'sortBy=publishedAt&'
        f'apiKey={API_KEY}'
    )

    response = requests.get(url)
    page_articles = response.json().get('articles', [])
    if not page_articles:
        break

    articles.extend(
        [
            article
            for article in page_articles
            if keyword.lower() in (article.get('title') or '').lower()
            or keyword.lower() in (article.get('description') or '').lower()
        ]
    )

    page += 1

total_score = 0
num_articles = 0

for i, article in enumerate(articles):
    print(f'Title: {article["title"]}')
    print(f'Published: {article["publishedAt"]}')
    print(f'Link: {article["url"]}')
    print(f'Description: {article["description"]}')

    content = article.get('content') or article.get('description') or article.get('title') or ''
    sentiment = pipe(content)[0]

    label = sentiment['label']
    confidence = sentiment['score']
    sentiment_value = confidence if label == 'positive' else -confidence if label == 'negative' else 0.0

    print(f'Sentiment {label}, Confidence: {confidence}, Value: {sentiment_value}')
    print('-' * 50)

    total_score += sentiment_value
    num_articles += 1

final_score = total_score / num_articles if num_articles else 0
print(f'Overall Sentiment: {"Positive" if final_score >= 0.15 else "Negative" if final_score <= -0.15 else "Neutral"} {final_score}')