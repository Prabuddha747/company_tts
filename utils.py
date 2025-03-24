import requests
from bs4 import BeautifulSoup
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import string
import re

# Download necessary NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def fetch_article_content(url):
    """
    Attempt to fetch and parse the full text of an article from its URL.
    Falls back to just the summary if full extraction fails.
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Try to find the main article content
        # This is a simplified approach - real sites require more specific extraction logic
        article_text = ""
        
        # Look for common article content containers
        article_tags = soup.find_all(['article', 'div', 'section'], class_=re.compile(r'article|content|story|entry'))
        
        if article_tags:
            # Get the largest content block (likely the main article)
            article_tag = max(article_tags, key=lambda x: len(x.get_text()))
            
            # Extract paragraphs
            paragraphs = article_tag.find_all('p')
            article_text = ' '.join([p.get_text().strip() for p in paragraphs])
        
        if not article_text:
            # Fallback to any paragraph text
            paragraphs = soup.find_all('p')
            article_text = ' '.join([p.get_text().strip() for p in paragraphs])
        
        # Clean the text
        article_text = re.sub(r'\s+', ' ', article_text).strip()
        
        return article_text if article_text else "Could not extract article content."
        
    except Exception as e:
        return f"Error extracting content: {str(e)}"

def summarize_text(text, num_sentences=3):
    """
    Create a simple extractive summary by selecting the most important sentences.
    Uses frequency of words to determine sentence importance.
    """
    # Tokenize and preprocess
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text.lower())
    
    # Remove punctuation and stopwords
    words = [word for word in words if word not in string.punctuation and word not in stop_words]
    
    # Calculate word frequencies
    freq_table = {}
    for word in words:
        if word in freq_table:
            freq_table[word] += 1
        else:
            freq_table[word] = 1
    
    # Score sentences based on word frequencies
    sentences = sent_tokenize(text)
    sentence_scores = {}
    
    for i, sentence in enumerate(sentences):
        for word in word_tokenize(sentence.lower()):
            if word in freq_table:
                if i in sentence_scores:
                    sentence_scores[i] += freq_table[word]
                else:
                    sentence_scores[i] = freq_table[word]
    
    # Get top sentences
    ranked_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)
    top_sentences = sorted(ranked_sentences[:num_sentences], key=lambda x: x[0])
    
    # Reconstruct summary
    summary = ' '.join([sentences[i] for i, _ in top_sentences])
    
    return summary

def clean_and_normalize_text(text):
    """
    Clean and normalize text for better analysis.
    """
    # Remove HTML tags if any
    text = re.sub(r'<.*?>', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove special characters
    text = re.sub(r'[^\w\s]', '', text)
    
    return text

def create_comparison_table(articles):
    """
    Create a pandas DataFrame for easy comparison of articles.
    """
    data = []
    for article in articles:
        data.append({
            'Title': article['Title'],
            'Sentiment': article['Sentiment'],
            'Topics': ', '.join(article['Topics']),
            'Date': article['Published Date']
        })
    
    return pd.DataFrame(data)