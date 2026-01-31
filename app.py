import feedparser
from transformers import pipeline

ticker = 'SI=F'
keyword = 'silver'

pipe = pipeline("text-classification", model="ProsusAI/finbert")

rss_url = f'https://finance.yahoo.com/rss/headline?s={ticker}'

feed = feedparser.parse(rss_url)

total_score = 0
num_articles = 0

for i, entry in enumerate(feed.entries):
    if keyword.lower() not in entry.summary.lower():
        continue

    print(f'Title: {entry.title}')
    print(f'Link: {entry.link}')
    print(f'Published: {entry.published}')
    print(f'Summary: {entry.summary}')

    sentiment = pipe(entry.summary)[0]
    label = sentiment['label']
    confidence = sentiment['score']
    sentiment_value = confidence if label == 'positive' else -confidence if label == 'negative' else 0.0
    print(f'Sentiment {label}, Confidence: {confidence}, Value: {sentiment_value}')
    print('-' * 50)

    total_score += sentiment_value
    num_articles += 1

final_score = total_score / num_articles if num_articles else 0
print(f'Overall Sentiment: {"Positive" if final_score >= 0.15 else "Negative" if final_score <= -0.15 else "Neutral"} {final_score}')