import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
import re
import os
import pandas as pd

def google_search(query, api_key, cse_id, num=2):
    try:
        url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={api_key}&cx={cse_id}&num={num}"
        response = requests.get(url)
        results = response.json().get('items', [])
        return [item.get('link') for item in results]
    except Exception as e:
        print(f"Error in google_search: {e}")
        return []

def clean_text(content):
    cleaned = re.sub(r'[!@#$%^&*()_+\-=\[\]{};:"\\|<>/]', ' ', content)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

def extract_content_without_noise(soup):
    for unwanted_tag in soup.find_all(['header', 'footer', 'nav', 'aside', 'script', 'style', 'figure', 'img']):
        unwanted_tag.decompose()
    tags_to_extract = ['p', 'div', 'article', 'section', 'span']
    content_fragments = [' '.join(tag.get_text().split()) for tag_name in tags_to_extract for tag in soup.find_all(tag_name) if len(tag.get_text().strip()) > 150]
    return clean_text(' '.join(content_fragments))

def get_relevant_content_from_url(url, query):
    try:
        response = requests.get(url, verify=False)
        soup = BeautifulSoup(response.text, 'html.parser')
        content = extract_content_without_noise(soup)
        vectorizer = CountVectorizer().fit_transform([query, content])
        vectors = vectorizer.toarray()
        cosine_sim = vectors[0] @ vectors.T
        if len(cosine_sim) > 1 and cosine_sim[1] > 0:
            return content
        else:
            return "No relevant content found or content is too dissimilar."
    except Exception as e:
        print(f"Error in get_relevant_content_from_url for {url}: {e}")
        return "Error occurred while fetching content."

def perform_search(query):
    API_KEY = "AIzaSyCr65UODhVzTpd6CIAFLDXQFyKbPXtZ0XY" # Replace with your API Key
    CSE_ID = "54b09b67591db427c" # Replace with your CSE ID

    try:
        print('================================================================')
        print('Câu hỏi là: ',query)
        print('================================================================')

        search_results = google_search(query, API_KEY, CSE_ID)
        contents = []
        for url in search_results:
            content = get_relevant_content_from_url(url, query)
            print('Content là: ',content)
            contents.append({
                "url": url,
                "Context": content
            })

        # File operations with try-catch
        try:
            # Delete the existing 'output.csv' file if it exists
            if os.path.exists('output.csv'):
                os.remove('output.csv')
            # Save to CSV
            df = pd.DataFrame(contents)
            df.to_csv('output.csv', index=False)
        except Exception as e:
            print(f"File operation error: {e}")

        return contents
    except Exception as e:
        print(f"Error in perform_search: {e}")
        return []

