import streamlit as st
import openai
from textblob import TextBlob
import pandas as pd

# Add your OpenAI API key
openai.api_key = 'your_openai_api_key'

# Function to generate a response from OpenAI
def generate_response(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        return "Sorry, I'm having trouble processing that request right now."

# Sentiment analysis function
def analyze_sentiment(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    if polarity > 0.5:
        return "Very Positive", polarity
    elif 0.1 < polarity <= 0.5:
        return "Positive", polarity
    elif -0.1 <= polarity <= 0.1:
        return "Neutral", polarity
    elif -0.5 < polarity < -0.1:
        return "Negative", polarity
    else:
        return "Very Negative", polarity

# Coping strategy function
def provide_coping_strategy(sentiment):
    strategies = {
        "Very Positive": "Keep up the positive vibes! Share your good mood with others.",
        "Positive": "It's great to see you're feeling positive. Keep it up!",
        "Neutral": "Feeling neutral is okay. Try an activity you enjoy.",
        "Negative": "If you're feeling down, take a break and relax.",
        "Very Negative": "Consider talking to a friend or seeking professional help."
    }
    return strategies.get(sentiment, "Stay positive, you're doing great!")

# Streamlit app title
st.title("Mental Health Support Chatbot")

# Initialize session state for messages and mood tracking
if 'messages' not in st.session_state:
    st.session_state['messages'] = []
if 'mood_tracker' not in st.session_state:
    st.session_state['mood_tracker'] = []

# Form for chat input
with st.form(key='chat_form'):
    user_message = st.text_input("You:")
    submit_button = st.form_submit_button(label='Send')

# Process user input on submit
if submit_button and user_message:
    st.session_state['messages'].append(("You", user_message))
    
    # Sentiment analysis and coping strategy
    sentiment, polarity = analyze_sentiment(user_message)
    coping_strategy = provide_coping_strategy(sentiment)
    
    # Generate response from GPT-3.5
    response = generate_response(user_message)
    
    # Append bot response and mood tracking to session state
    st.session_state['messages'].append(("Bot", response))
    st.session_state['mood_tracker'].append((user_message, sentiment, polarity))

    # Display coping strategy
    st.write(f"Suggested Coping Strategy: {coping_strategy}")

# Display chat messages
for sender, message in st.session_state['messages']:
    if sender == "You":
        st.text(f"You: {message}")
    else:
        st.text(f"Bot: {message}")

# Mood tracking chart
if st.session_state['mood_tracker']:
    mood_data = pd.DataFrame(st.session_state['mood_tracker'], columns=["Message", "Sentiment", "Polarity"])
    st.line_chart(mood_data['Polarity'])

# Sidebar resources and session summary
st.sidebar.title("Resources")
st.sidebar.write("If you need immediate help, contact these resources:")
st.sidebar.write("1. National Suicide Prevention Lifeline: 1-800-273-8255")
st.sidebar.write("2. Crisis Text Line: Text 'HELLO' to 741741")
st.sidebar.write("[More Resources](https://www.mentalhealth.gov/get-help/immediate-help)")

if st.sidebar.button("Show Session Summary"):
    st.sidebar.write("### Session Summary")
    for i, (message, sentiment, polarity) in enumerate(st.session_state['mood_tracker']):
        st.sidebar.write(f"{i+1}. {message} - Sentiment: {sentiment} (Polarity: {polarity})")
