import streamlit as st
import pandas as pd
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Function to analyze sentiment using TextBlob
def analyze_sentiment_textblob(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

# Function to analyze sentiment using VADER
def analyze_sentiment_vader(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = analyzer.polarity_scores(text)
    if sentiment_scores['compound'] > 0.05:
        return "Positive"
    elif sentiment_scores['compound'] < -0.05:
        return "Negative"
    else:
        return "Neutral"

# Streamlit app title
st.title("Real-Time Sentiment Analysis and Social Media Data Visualization")

# Step 1: Data Loading
st.header("Uploaded Dataset Overview")
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Load and clean data
    df = pd.read_csv(uploaded_file)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df.drop(columns=["Unnamed: 0.1", "Unnamed: 0"], errors="ignore", inplace=True)
    df.fillna("", inplace=True)
    
    # Show dataset information
    st.write("**Dataset Overview:**")
    st.dataframe(df.head())

    # Display basic stats
    st.write("**Basic Dataset Information:**")
    st.write(df.describe(include="all"))

    # Step 2: Text Input for Real-Time Sentiment Analysis
    st.header("Real-Time Sentiment Analysis")
    user_input = st.text_area("Enter text for sentiment analysis:")
    if st.button("Analyze Sentiment"):
        if user_input:
            textblob_sentiment = analyze_sentiment_textblob(user_input)
            vader_sentiment = analyze_sentiment_vader(user_input)

            st.write(f"**TextBlob Sentiment:** {textblob_sentiment}")
            st.write(f"**VADER Sentiment:** {vader_sentiment}")
        else:
            st.write("Please enter text for analysis.")

    # Step 3: Data Visualizations
    st.header("Data Visualizations")

    # Sentiment Distribution Bar Chart
    st.subheader("Sentiment Distribution")
    if 'Sentiment' in df.columns:
        sentiment_counts = df['Sentiment'].value_counts()
        st.bar_chart(sentiment_counts)

    # Platform Distribution Bar Chart
    st.subheader("Platform Distribution")
    if 'Platform' in df.columns:
        platform_counts = df['Platform'].value_counts()
        st.bar_chart(platform_counts)

    # Hashtag Count Bar Chart
    st.subheader("Top 10 Hashtags by Count")
    if 'Hashtags' in df.columns:
        top_hashtags = df['Hashtags'].value_counts().head(10)
        st.bar_chart(top_hashtags)

    # Total Likes by Year Line Chart
    st.subheader("Total Likes by Year")
    if 'Year' in df.columns and 'Likes' in df.columns:
        likes_by_year = df.groupby("Year")["Likes"].sum().reset_index()
        st.line_chart(likes_by_year.set_index("Year"))

    # Likes vs Hour of Day Bar Chart
    st.subheader("Likes vs Hour of Day")
    if 'Hour' in df.columns and 'Likes' in df.columns:
        likes_by_hour = df.groupby("Hour")["Likes"].sum()
        st.bar_chart(likes_by_hour)

else:
    st.write("Please upload a CSV file to proceed.")
