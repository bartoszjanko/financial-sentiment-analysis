import os
import requests
from dotenv import load_dotenv

language = 'en'

load_dotenv()

API_KEY = os.getenv('key')
if not API_KEY:
    raise ValueError('Missing API_KEY in environment.')

params = {
    'language': language,
    'apiKey': API_KEY,
}

response = requests.get('https://newsapi.org/v2/sources', params=params)
response.raise_for_status()

sources_data = response.json().get('sources', [])

for source in sources_data:
    print(f"{source.get('id')}: {source.get('name')} ({source.get('url')})")
