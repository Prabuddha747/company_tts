# News Sentiment Analysis & Text-to-Speech Application

This web application extracts key details from news articles related to a given company, performs sentiment analysis, conducts comparative analysis, and generates a text-to-speech (TTS) output in Hindi.

## Features

- **News Extraction**: Scrapes and displays information from 10+ unique news articles related to the given company using BeautifulSoup.
- **Sentiment Analysis**: Analyzes the sentiment of each article (positive, negative, neutral).
- **Comparative Analysis**: Compares sentiment across articles to derive insights on news coverage variations.
- **Text-to-Speech**: Converts summarized content into Hindi speech using an open-source TTS model.
- **User Interface**: Simple web-based interface using Streamlit.
- **API Backend**: Communication between frontend and backend via RESTful APIs.

## Technology Stack

- **Frontend**: Streamlit
- **Backend API**: FastAPI
- **Scraping**: BeautifulSoup, Requests
- **NLP & Analysis**: NLTK, Transformers, Scikit-learn
- **Translation & TTS**: MBart, gTTS
- **Deployment**: Hugging Face Spaces

## Setup Instructions

### Prerequisites

- Python 3.8+
- pip (Python package manager)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/news-sentiment-tts.git
   cd news-sentiment-tts
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download required NLTK resources:
   ```python
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('vader_lexicon')"
   ```

### Running the Application

1. Start the FastAPI backend server:
   ```bash
   uvicorn api:app --reload --port 8000
   ```

2. In a separate terminal, start the Streamlit frontend:
   ```bash
   streamlit run app.py
   ```

3. Open your browser and navigate to:
   - Frontend: http://localhost:8501
   - API Docs: http://localhost:8000/docs

## Usage

1. Enter a company name or select from the sample companies list.
2. Click "Analyze News" to start the analysis process.
3. View the results in the three tabs:
   - Summary: Overall sentiment analysis and key findings
   - Articles: Detailed information about each analyzed article
   - Audio: Hindi audio summary that can be played or downloaded

## API Endpoints

- `GET /`: Check if API is running
- `GET /api/companies`: Get list of sample companies
- `POST /api/analyze`: Analyze news for a specific company
- `GET /api/cached/{company_name}`: Get cached analysis results
- `POST /api/cache`: Cache analysis results

## Deployment

The application is deployed on Hugging Face Spaces and can be accessed at:
[https://huggingface.co/spaces/your-username/news-sentiment-tts](https://huggingface.co/spaces/your-username/news-sentiment-tts)

### Deploying to Hugging Face Spaces

1. Create a Hugging Face account if you don't have one.
2. Create a new Space and select Streamlit as the SDK.
3. Upload the code to the Space repository.
4. Edit the Space configuration to include both FastAPI and Streamlit.

## Assumptions & Limitations

- The application focuses on non-JS websites that can be scraped with BeautifulSoup.
- Some news sites may block scraping attempts or require authentication.
- Sentiment analysis accuracy depends on the pre-trained models and may not capture all nuances.
- Translation to Hindi may not be perfect, especially for technical terms.
- Rate limiting by search engines may affect the number of articles retrieved.

## Future Improvements

- Add support for more languages
- Implement more advanced NLP techniques for better topic extraction
- Improve caching mechanism for faster repeated queries
- Add user authentication for personalized dashboards
- Expand comparative analysis with historical data