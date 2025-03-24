import streamlit as st
import requests
import feedparser
from bs4 import BeautifulSoup
from textblob import TextBlob
from keybert import KeyBERT
import json
from deep_translator import GoogleTranslator
import os

# ✅ Google Gemini API Configuration
GOOGLE_API_KEY = "AIzaSyA0dKIWOSNnednOwlJW1JTWUW_wiTNcwww"
GEMINI_ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent?key={GOOGLE_API_KEY}"

# ✅ Initialize KeyBERT model for topic extraction
kw_model = KeyBERT()

# ✅ Function to clean HTML from news summaries
def clean_html(raw_html):
    """Removes HTML tags from text."""
    soup = BeautifulSoup(raw_html, "html.parser")
    return soup.get_text()

# ✅ Function to analyze sentiment
def analyze_sentiment(text):
    """Performs improved sentiment analysis using TextBlob."""
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    
    if polarity > 0.2 and subjectivity > 0.4:
        return "Positive"
    elif polarity < -0.2 and subjectivity > 0.4:
        return "Negative"
    else:
        return "Neutral"

# ✅ Function to extract key topics using KeyBERT
def extract_topics(text):
    """Extracts key topics from text using KeyBERT."""
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1,2), stop_words='english', top_n=3)
    return list(set(kw[0] for kw in keywords))

# ✅ Function to fetch news from Google News RSS
def get_google_news(company):
    """Fetches latest news articles from Google News RSS."""
    rss_url = f"https://news.google.com/rss/search?q={company}&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(rss_url)
    articles = []
    
    for entry in feed.entries[:10]:
        summary_cleaned = clean_html(entry.summary)
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

# ✅ Function to call Google Gemini API
def call_gemini_api(prompt):
    """Calls Google Gemini API for AI-based analysis."""
    headers = {"Content-Type": "application/json"}
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    
    try:
        response = requests.post(GEMINI_ENDPOINT, headers=headers, json=payload, timeout=20)
        response.raise_for_status()
        result = response.json()
        
        if "candidates" in result:
            return result["candidates"][0].get("content", "No response received.")
        else:
            return "⚠️ Error: Unexpected API response format."
    
    except requests.exceptions.RequestException as e:
        return f"❌ API Request Failed: {str(e)}"

# ✅ Function to generate comparative analysis using Gemini AI
def generate_comparative_analysis(articles):
    """Generates AI-driven comparative analysis of news articles."""
    prompt = f"""
    Given the following news articles, perform a comparative analysis:
    {json.dumps(articles, indent=4)}
    Identify common and unique topics, sentiment variations, and impact.
    Output should be JSON format.
    """
    return call_gemini_api(prompt)

# ✅ Function to generate final sentiment summary
def generate_final_sentiment(articles):
    """Generates AI-driven final sentiment analysis of news."""
    prompt = f"""
    Analyze the overall sentiment trend in the following news articles:
    {json.dumps(articles, indent=4)}
    Provide a concise final sentiment summary.
    """
    return call_gemini_api(prompt)

# ✅ Function to translate text to Hindi using Deep Translator
def translate_to_hindi(text):
    """Translates text to Hindi using GoogleTranslator."""
    try:
        return GoogleTranslator(source="auto", target="hi").translate(text)
    except Exception as e:
        return f"⚠️ Translation Error: {e}"

# ✅ Streamlit UI
st.title("📢 News Summarization & Sentiment Analysis")
company_name = st.text_input("Enter Company Name:")

if st.button("Analyze News"):
    if company_name:
        articles = get_google_news(company_name)
        
        if not articles:
            st.error("❌ No news articles found.")
        else:
            # ✅ Display latest news
            st.subheader("📰 Latest News Articles")
            for article in articles:
                st.write(f"**{article['Title']}**")
                st.write(f"📖 {article['Summary']}")
                st.write(f"💡 Topics: {', '.join(article['Topics'])}")
                st.write(f"📅 {article['Published Date']}")
                st.write(f"🔗 [Read More]({article['Link']})")
                st.write(f"📊 Sentiment: {article['Sentiment']}")
                st.write("---")
            
            # ✅ Generate and display comparative analysis
            st.subheader("📊 Comparative Analysis")
            comparative_analysis = generate_comparative_analysis(articles)
            st.write(comparative_analysis)
            
            # ✅ Generate and display final sentiment analysis
            st.subheader("📢 Final Sentiment Analysis")
            final_sentiment = generate_final_sentiment(articles)
            st.write(final_sentiment)
            
            # ✅ Translate sentiment analysis to Hindi
            st.subheader("🗣️ Hindi Text-to-Speech (TTS)")
            hindi_translation = translate_to_hindi(final_sentiment)
            st.write(hindi_translation)
    
    else:
        st.error("⚠️ Please enter a company name.")
