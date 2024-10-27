import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
from textblob import TextBlob
import spacy
import re
from collections import defaultdict
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    import os
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Initialize session state
if 'entries' not in st.session_state:
    st.session_state.entries = []

class SentimentAnalyzer:
    def __init__(self):
        # Enhanced emotion lexicon
        self.positive_words = {
            'happy': 1, 'joy': 1, 'excited': 1, 'grateful': 1, 'peaceful': 0.8,
            'calm': 0.6, 'good': 0.5, 'better': 0.6, 'blessed': 0.8, 'wonderful': 1,
            'fantastic': 1, 'great': 0.8, 'pleasant': 0.6, 'satisfied': 0.7,
            'confident': 0.8, 'motivated': 0.8, 'optimistic': 0.7, 'proud': 0.7
        }
        
        self.negative_words = {
            'sad': -1, 'angry': -1, 'depressed': -1, 'anxious': -0.8, 'worried': -0.7,
            'stressed': -0.8, 'frustrated': -0.7, 'upset': -0.6, 'terrible': -0.9,
            'horrible': -1, 'miserable': -0.9, 'lonely': -0.7, 'tired': -0.5,
            'exhausted': -0.8, 'afraid': -0.7, 'hopeless': -0.9, 'worthless': -1
        }
        
        # Intensity modifiers
        self.intensifiers = {
            'very': 1.5, 'extremely': 2.0, 'really': 1.5, 'so': 1.3,
            'absolutely': 2.0, 'completely': 1.8, 'totally': 1.7,
            'utterly': 1.8, 'highly': 1.6, 'intensely': 1.7
        }
        
        self.diminishers = {
            'slightly': 0.5, 'somewhat': 0.7, 'kind of': 0.6, 'sort of': 0.6,
            'a bit': 0.5, 'little': 0.5, 'barely': 0.3, 'hardly': 0.3
        }
        
        # Negation words
        self.negations = {'not', 'no', "n't", 'never', 'none', 'nobody', 'nowhere', 'neither'}
        
    def get_window_score(self, tokens, index, window_size=3):
        """Calculate score based on surrounding words"""
        start = max(0, index - window_size)
        end = min(len(tokens), index + window_size + 1)
        
        score = 0
        for i in range(start, end):
            if i == index:
                continue
            word = tokens[i].lower()
            if word in self.intensifiers:
                score += 0.2
            elif word in self.diminishers:
                score -= 0.2
        return score
    
    def check_negation(self, tokens, index, window_size=3):
        """Check for negation words in the vicinity"""
        start = max(0, index - window_size)
        for i in range(start, index):
            if tokens[i].lower() in self.negations:
                return True
        return False
    
    def analyze_sentiment(self, text):
        """Enhanced sentiment analysis with contextual understanding"""
        # Initial TextBlob analysis for baseline
        blob = TextBlob(text)
        initial_polarity = blob.sentiment.polarity
        
        # Tokenization and preprocessing
        tokens = word_tokenize(text.lower())
        stops = set(stopwords.words('english')) - self.negations
        
        # Detailed analysis
        sentiment_score = 0
        word_count = 0
        
        for i, token in enumerate(tokens):
            token = token.lower()
            if token in stops:
                continue
                
            # Check base sentiment
            score = 0
            if token in self.positive_words:
                score = self.positive_words[token]
            elif token in self.negative_words:
                score = self.negative_words[token]
                
            if score != 0:
                # Apply contextual modifications
                is_negated = self.check_negation(tokens, i)
                if is_negated:
                    score *= -1
                    
                # Consider surrounding intensifiers/diminishers
                context_score = self.get_window_score(tokens, i)
                score *= (1 + context_score)
                
                sentiment_score += score
                word_count += 1
        
        # Combine both analyses
        if word_count > 0:
            final_score = (sentiment_score / word_count + initial_polarity) / 2
        else:
            final_score = initial_polarity
            
        # Enhanced classification logic
        if final_score > 0.1:
            strength = min(abs(final_score), 1)
            return 'positive', final_score, strength
        elif final_score < -0.1:
            strength = min(abs(final_score), 1)
            return 'negative', final_score, strength
        else:
            return 'neutral', final_score, 0.5

class MentalHealthAnalyzer:
    def __init__(self):
        self.sentiment_analyzer = SentimentAnalyzer()
        self.concern_categories = {
            'anxiety': ['anxious', 'worried', 'nervous', 'panic', 'fear'],
            'depression': ['depressed', 'sad', 'hopeless', 'empty', 'worthless'],
            'stress': ['stressed', 'overwhelmed', 'pressure', 'tension'],
            'insomnia': ['sleep', 'insomnia', 'tired', 'exhausted', 'fatigue'],
            'eating_disorder': ['eating', 'food', 'weight', 'appetite']
        }

    def get_polarity(self, text):
        """Analyze sentiment polarity of text using enhanced analyzer"""
        sentiment, polarity, strength = self.sentiment_analyzer.analyze_sentiment(text)
        return sentiment, polarity, strength

    def extract_keywords(self, text):
        """Extract mental health-related keywords using spaCy"""
        doc = nlp(text.lower())
        keywords = []
        
        # Extract noun phrases and emotional terms
        for chunk in doc.noun_chunks:
            keywords.append(chunk.text)
        
        # Add individual tokens that might be relevant
        keywords.extend([token.text for token in doc 
                        if token.pos_ in ['ADJ', 'VERB'] 
                        and not token.is_stop])
        
        return list(set(keywords))

    def classify_concerns(self, keywords):
        """Classify concerns into predefined categories"""
        concerns = defaultdict(float)
        
        for keyword in keywords:
            for category, terms in self.concern_categories.items():
                if any(term in keyword for term in terms):
                    concerns[category] += 1
                    
        return dict(concerns)

    def calculate_intensity(self, text):
        """Calculate intensity score based on linguistic cues"""
        intensity_markers = {
            'extremely': 3,
            'very': 2,
            'really': 2,
            'severely': 3,
            'slightly': -1,
            'little': -1,
            'mildly': -1
        }
        
        base_score = 5
        text_lower = text.lower()
        
        # Adjust score based on intensity markers
        for marker, value in intensity_markers.items():
            if marker in text_lower:
                base_score += value
                
        # Adjust for exclamation marks and capitalization
        base_score += text.count('!') * 0.5
        caps_ratio = sum(1 for c in text if c.isupper()) / len(text)
        base_score += caps_ratio * 2
        
        return min(max(base_score, 1), 10)

def main():
    st.set_page_config(page_title="Mental Health Analyzer", layout="wide")
    
    st.title("Mental Health Analysis Dashboard")
    st.markdown("""
    This application helps track and analyze mental health patterns over time.
    Share your thoughts and feelings to get insights about your emotional well-being.
    """)

    analyzer = MentalHealthAnalyzer()

    # Input section
    with st.container():
        st.subheader("Share Your Thoughts")
        user_input = st.text_area("How are you feeling today?", height=100)
        
        if st.button("Analyze"):
            if user_input:
                # Perform analysis with enhanced sentiment analysis
                sentiment, polarity, strength = analyzer.get_polarity(user_input)
                keywords = analyzer.extract_keywords(user_input)
                concerns = analyzer.classify_concerns(keywords)
                intensity = analyzer.calculate_intensity(user_input)
                
                # Store the entry with enhanced sentiment data
                entry = {
                    'timestamp': datetime.now(),
                    'text': user_input,
                    'sentiment': sentiment,
                    'polarity': polarity,
                    'sentiment_strength': strength,
                    'keywords': keywords,
                    'concerns': concerns,
                    'intensity': intensity
                }
                st.session_state.entries.append(entry)

    # Analysis Display
    if st.session_state.entries:
        latest_entry = st.session_state.entries[-1]
        
        # Current Analysis with enhanced sentiment display
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Current Mood Analysis")
            sentiment_color = {
                'positive': 'green',
                'neutral': 'gray',
                'negative': 'red'
            }[latest_entry['sentiment']]
        with col2:
            st.subheader("Detected Concerns")
            if latest_entry['concerns']:
                for category, count in latest_entry['concerns'].items():
                    st.write(f"• {category.replace('_', ' ').title()}")
            else:
                st.write("No specific concerns detected")

        with col3:
            st.subheader("Key Phrases")
            st.write(", ".join(latest_entry['keywords'][:5]))

            
            # Enhanced sentiment display with strength indicator
            st.markdown(f"""
                **Sentiment:** <span style='color:{sentiment_color}'>
                {latest_entry['sentiment'].title()} 
                (Strength: {latest_entry['sentiment_strength']:.2f})
                </span>
                """, unsafe_allow_html=True)
            st.metric("Intensity Score", f"{latest_entry['intensity']:.1f}/10")

        # Historical Analysis
        st.subheader("Mood Trends Over Time")
        
        # Prepare data for plotting
        df = pd.DataFrame([{
            'Date': entry['timestamp'],
            'Intensity': entry['intensity'],
            'Polarity': entry['polarity']
        } for entry in st.session_state.entries])
        
        # Plot trends
        fig = px.line(df, x='Date', y=['Intensity', 'Polarity'],
                     title='Mental Health Trends Over Time')
        st.plotly_chart(fig, use_container_width=True)

        # Concern Distribution
        st.subheader("Concern Distribution")
        all_concerns = defaultdict(int)
        for entry in st.session_state.entries:
            for category in entry['concerns']:
                all_concerns[category] += 1
                
        if all_concerns:
            fig = px.pie(values=list(all_concerns.values()),
                        names=[c.replace('_', ' ').title() for c in all_concerns.keys()],
                        title='Distribution of Concerns')
            st.plotly_chart(fig, use_container_width=True)

        # Historical Entries
        st.subheader("Previous Entries")
        for entry in reversed(st.session_state.entries[:-1]):
            with st.expander(f"Entry from {entry['timestamp'].strftime('%Y-%m-%d %H:%M')}"):
                st.write(entry['text'])
                st.write(f"Sentiment: {entry['sentiment']}")
                st.write(f"Intensity: {entry['intensity']:.1f}/10")

if __name__ == "__main__":
    main()
