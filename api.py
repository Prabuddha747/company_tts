import requests
import feedparser
from bs4 import BeautifulSoup
from textblob import TextBlob
from keybert import KeyBERT
import json
import tempfile
from gtts import gTTS
import os

# Initialize KeyBERT model for topic extraction
kw_model = KeyBERT()

# Load Google Gemini API Key from environment variable
GOOGLE_API_KEY = "AIzaSyA0dKIWOSNnednOwlJW1JTWUW_wiTNcwww"
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is missing. Set it as an environment variable.")

GEMINI_ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent?key={GOOGLE_API_KEY}"

def clean_html(raw_html):
    """Removes HTML tags from text."""
    soup = BeautifulSoup(raw_html, "html.parser")
    return soup.get_text()

def analyze_sentiment(text):
    """Performs improved sentiment analysis using TextBlob."""
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity  # -1 (negative) to +1 (positive)
    subjectivity = blob.sentiment.subjectivity  # 0 (objective) to 1 (subjective)

    # Adjust based on subjectivity (more subjective = stronger sentiment)
    if polarity > 0.2 and subjectivity > 0.4:
        return "Positive"
    elif polarity < -0.2 and subjectivity > 0.4:
        return "Negative"
    else:
        return "Neutral"

def extract_topics(text):
    """Extracts key topics from text using KeyBERT."""
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=3)
    return list(set(kw[0] for kw in keywords))  # Ensure unique topics

def get_google_news(company):
    """
    Fetches the latest 10 news articles about the given company from Google News RSS.
    Cleans summary text, analyzes sentiment, and extracts key topics.
    """
    rss_url = f"https://news.google.com/rss/search?q={company}&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(rss_url)

    articles = []
    for entry in feed.entries[:10]:  # Get first 10 articles
        summary_cleaned = clean_html(entry.summary) if hasattr(entry, "summary") else "No summary available"
        sentiment = analyze_sentiment(summary_cleaned)
        topics = extract_topics(summary_cleaned)

        articles.append({
            "Title": entry.title,
            "Summary": summary_cleaned,
            "Sentiment": sentiment,
            "Topics": topics,
            "Published Date": entry.published if hasattr(entry, 'published') else "Unknown",
            "Link": entry.link
        })

    return articles

def call_gemini_api(prompt):
    """Calls Google Gemini API with error handling and proper response parsing."""
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.2,
            "topP": 0.8,
            "topK": 40
        }
    }

    try:
        response = requests.post(GEMINI_ENDPOINT, headers=headers, json=payload, timeout=20)
        response.raise_for_status()
        result = response.json()

        # Handle API response structure changes
        candidates = result.get("candidates", [])
        if candidates and "content" in candidates[0]:
            parts = candidates[0]["content"].get("parts", [])
            return parts[0].get("text", "No response from AI.") if parts else "No response from AI."

        return "Error: Unexpected API response format."

    except requests.exceptions.Timeout:
        return "Error: API request timed out."
    except requests.exceptions.RequestException as e:
        return f"Error: API Request Failed - {str(e)}"

def generate_comparative_analysis(articles):
    """
    Uses Google Gemini AI to generate a detailed comparative analysis.
    Returns formatted text without JSON structure.
    """
    prompt = f"""
    Given the following news articles, perform a detailed comparative analysis:

    {json.dumps(articles, indent=4)}

    1. Identify common and unique topics.
    2. Compare sentiment variations across articles.
    3. Provide a structured comparison, highlighting how different articles present the company.
    4. Explain the impact of these differences on investors and public perception.

    Output should be formatted with Markdown and structured as:
    - Coverage Differences (Comparison & Impact)
    - Topic Overlap (Common & Unique Topics)

    DO NOT include any JSON or XML formatting in the response, just plain text with Markdown.
    """

    return call_gemini_api(prompt)

def generate_final_sentiment(articles):
    """
    Uses Google Gemini AI to generate a final sentiment summary based on news articles.
    Returns formatted text without JSON structure.
    """
    prompt = f"""
    Analyze the overall sentiment trend in the following news articles:

    {json.dumps(articles, indent=4)}

    Provide a well-reasoned final sentiment analysis explaining:
    - The general tone of the news coverage.
    - Whether it is likely to have a positive or negative impact on investors.
    - Any risks or opportunities presented in the articles.

    Output should be in Markdown format, structured into:
    - Overall Sentiment
    - Risks
    - Opportunities
    
    DO NOT include any JSON or XML formatting in the response, just plain text with Markdown.
    """

    return call_gemini_api(prompt)

def get_text_to_speech(text):
    """
    Converts text to speech in Hindi and returns the audio file path.
    """
    # Create a temporary file path
    temp_file_path = tempfile.mktemp(suffix=".mp3")
    
    # Generate Hindi audio
    tts = gTTS(text=text, lang='hi', slow=False)
    tts.save(temp_file_path)
    
    return temp_file_path

def process_company_news(company_name):
    """
    Main function to process company news and return structured results.
    This is the API endpoint that combines all functionality.
    """
    # Get news articles
    articles = get_google_news(company_name)
    
    if not articles:
        return {"error": "No news articles found. Try another company name."}
    
    # Generate comparative analysis and final sentiment
    comparative_analysis = generate_comparative_analysis(articles)
    final_sentiment = generate_final_sentiment(articles)
    
    # Build structured output
    structured_output = {
        "Company": company_name,
        "Articles": articles,
        "Comparative Sentiment Score": comparative_analysis,
        "Final Sentiment Analysis": final_sentiment,
        "Audio": "[Play Hindi Speech]"  # Placeholder replaced in the frontend
    }
    
    return structured_output
